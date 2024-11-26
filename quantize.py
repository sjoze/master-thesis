import os
import torch
import tensorrt
import onnx
import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3) # ERROR
from onnxruntime.quantization import (
    CalibrationDataReader,
    create_calibrator,
    write_calibration_table,
    CalibrationMethod,
)
from torch.utils.data import DataLoader
from statistics import mean
from helper import calculate_accuracy
from torch.profiler import profile, record_function, ProfilerActivity


N = 1000

# Calibration Dataset for ONNX to prepare model for TensorRT
class ONNXCalibrationDataset(CalibrationDataReader):
    def __init__(self, C, W, H, batch_size=128):
        super().__init__()
        self.count = 0
        self.total = N
        self.batch_size = batch_size
        self.C = C
        self.W = W
        self.H = H

    def get_next(self) -> dict:
        if self.count < self.total:
            self.count += 1
            return {"input": torch.randn(self.batch_size, self.C, self.H, self.W).numpy()}
        else:
            return None


# Computing quantization parameters on a small set of training or evaluation data and export the resulting model and a calibration file
def prepare(model, trt_path, dataset, batch_size=128):
    C, W, H = dataset.__getitem__(0)[0].shape
    cwd = os.getcwd()
    os.chdir(trt_path)
    device = torch.device("cuda")
    sample = torch.randn(batch_size, C, H, W).to(device)
    torch.onnx.export(
        model,
        args=sample,
        f="model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
    )
    model = onnx.load("model.onnx")
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, "model_dynamic.onnx")

    # Create calibration dataset with dummy data
    calibration_dataset = ONNXCalibrationDataset(batch_size=batch_size, C=C, W=W, H=H)
    calibrator = create_calibrator(
        model="model_dynamic.onnx",
        op_types_to_calibrate=[],
        augmented_model_path="model_augmented.onnx",
        calibrate_method=CalibrationMethod.MinMax           # alt: Entropy, Percentile, Distribution
    )
    calibrator.set_execution_providers(["CUDAExecutionProvider"])
    calibrator.collect_data(data_reader=calibration_dataset)

    compute_data = calibrator.compute_data().data
    new_compute_data = {k: (float(v.range_value[0]), float(v.range_value[1])) for k, v in compute_data.items()}
    # Export calibration results
    write_calibration_table(new_compute_data)
    os.chdir(cwd)


# Execute TensorRT engine
def run_tensorrt(dataset, trt_path, inference_type="int8", use_trt_cache=True, batch_size=128):

    C, W, H = dataset.__getitem__(0)[0].shape
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 12
    sess_options.inter_op_num_threads = 12

    print(f'########## [TensorRT - {"FP16" if inference_type == "fp16" else "INT8" if inference_type == "int8" else "BASE"}] Model loading...')
    model = ort.InferenceSession(
        # Define model to run
        path_or_bytes= trt_path + "/model_dynamic.onnx",
        sess_options=sess_options,
        providers=[
            (
                "TensorrtExecutionProvider",
                {
                    'trt_engine_cache_enable': use_trt_cache,
                    'trt_engine_cache_path': os.getcwd() +'/data/trt_caches',
                    } | \
                {
                    # Use INT8 if possible, fall back to FP16 otherwise
                    "trt_fp16_enable": True,
                    "trt_int8_enable": True,
                    "trt_int8_calibration_table_name": "calibration.flatbuffers",
                } if inference_type == 'int8' else \
                {
                    # Use FP16
                    'trt_engine_cache_enable': use_trt_cache,
                    'trt_engine_cache_path': os.getcwd() +'/data/trt_caches',
                    "trt_fp16_enable": True,
                } if inference_type == "fp16" else \
                {
                    # Run without any quantization
                    'trt_engine_cache_enable': use_trt_cache,
                    'trt_engine_cache_path': os.getcwd() +'/data/trt_caches',
                    "trt_int8_enable": False,
                    "trt_fp16_enable": False,
                },
            ),
            "CUDAExecutionProvider",
        ],
    )


    print(f'########## [TensorRT - {"FP16" if inference_type == "fp16" else "INT8" if inference_type == "int8" else "BASE"}] Model loading Done!')

    # Warumup with dummy inference
    print(f'########## [TensorRT - {"FP16" if inference_type == "fp16" else "INT8" if inference_type == "int8" else "BASE"}] Optimization and Warmup process...')
    for i in range(10):
        data = np.ones([batch_size, C, H, W], dtype=np.float32)
        _ = model.run(["output"], {"input": data})
    print(f'########## [TensorRT - {"FP16" if inference_type == "fp16" else "INT8" if inference_type == "int8" else "BASE"}] Optimization and Warmup Done!')
    print(f'########## [TensorRT - {"FP16" if inference_type == "fp16" else "INT8" if inference_type == "int8" else "BASE"}] Inference started...')
    accs = []
    inference_times = []
    for x, y in dataloader:
        with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                output = model.run(["output"], {'input': x.numpy()})
        for event in prof.key_averages():
            if event.key == "model_inference":
                inference_times.append(event.cuda_time/1000)
        acc = calculate_accuracy(dataset, output, y, "q")
        accs.append(acc)

    print(f'########## [TensorRT - {"FP16" if inference_type == "fp16" else "INT8"}] Inference DONE')
    print("########## Accuracy: " + str(mean(accs)))
    print(f'########## [TensorRT - {"FP16" if inference_type == "fp16" else "INT8"}] Batch Inference on AVG: {mean(inference_times)} ms')

    # Calculate "model" size of the TensorRT engine
    engine_file_size = 0
    files = [f for f in os.listdir(trt_path) if '.engine' in f]
    if len(files) > 0:
        engine_file_size = os.path.getsize(os.path.join(trt_path, files[0]))/ (1024 * 1024)

    return {"accuracy": "%.2f" % (mean(accs)), "inf_time": mean(inference_times), "model_size": engine_file_size, "batch_size": batch_size}
