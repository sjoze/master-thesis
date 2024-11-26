import time
from statistics import mean
from torch.utils.data import DataLoader
import torch
import csv
from helper import get_model_size, ConfigReader, calculate_accuracy
from datetime import datetime
from quantize import prepare, run_tensorrt
from torch.profiler import profile, record_function, ProfilerActivity
import os
import itertools
from pathlib import Path


device = torch.device("cuda")
print(f"########## CUDNN IS AVAILABLE: {torch.backends.cudnn.is_available()}")
print("########## WORKING ON: " + torch.cuda.get_device_name())

# Placeholder variables for CSV file and config reader
f_csv = False
csv_writer = False
cfg = False

# Create experiment folder and experiment CSV file
def create_experiment_csv(exp_path, exp_file_name):
	# Experiment named after start time if not customized otherwise
	if exp_file_name == "":
		exp_file_name = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
	csv_fields = ["Model", "Pruner", "Quantizer", "Dataset", "Amount", "Accuracy", "Inference Time (ms)", "Original Size", "Compressed Size", "Batch Size", "Prep Time (s)"]
	Path(exp_path).mkdir(parents=True, exist_ok=True)
	global f_csv
	f_csv = open(exp_path +  "/Experiment_" + exp_file_name + ".csv", "a")
	global csv_writer
	csv_writer = csv.writer(f_csv)
	csv_writer.writerow(csv_fields)


"""
MODEL EXECUTION
"""
def run_model(model, dataset, batch_size):
	data_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
	accs = []
	inference_times = []
	for x, y in data_generator:
		x, y = x.to(device), y
		# Track GPU usage on inference
		with profile(activities=[
			ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
			with record_function("model_inference"):
				output = model(x)
		for event in prof.key_averages():
			if event.key == "model_inference":
				inference_times.append(event.cuda_time/1000)	# Lists time as ms but returns it as µs
		output = output.cpu().detach()
		acc = calculate_accuracy(dataset, output, y, "p")
		accs.append(acc)
	print("########## Accuracy: " + "%.2f%%" % (mean(accs)))
	print(f"########## Per Batch on AVG: {mean(inference_times)} ms")
	return {"accuracy": "%.2f" % (mean(accs)), "inf_time": mean(inference_times), "model_size": get_model_size(model), "batch_size": batch_size}


# Process some batches to warm up GPU
def warmup_model(model, iterations, batch_size, C, H, W):
	for i in range(iterations):
		dummy_data = torch.rand(batch_size, C, H, W).to(device)
		_ = model(dummy_data)


"""
MODEL COMPRESSION AND BENCHMARKING
"""
def benchmark_pq(model, dataset, batchsize, mode, pruner="-", quantizer="-", trt_path="./data/trt_caches", exp_path="./data/experiments", exp_file_name=""):
	# If this method is called on its own (i.e. with custom settings and not our benchmark config), create experiment file with specified file name
	if not f_csv:
		create_experiment_csv(exp_path, exp_file_name)
	model = model.eval().to(device)
	# If no pruner specified
	amount = pruner_name = "-"
	if "p" in mode:
		amount = pruner.get_amount()
		pruner_name = pruner.__str__()

	model_name, dataset_name = [model.__class__.__name__, dataset.__str__()]
	orig_model_size = get_model_size(model)
	C, W, H = dataset.__getitem__(0)[0].shape

	print("########## MODEL: % s with PRUNER: % s and QUANTIZER: % s and AMOUNT: % s on DATASET: %s with BATCHSIZE: %s" % (model_name, pruner_name, quantizer, amount, dataset_name, batchsize))
	start_time = time.perf_counter()
	# If pruning degree inside of (0,1], prune. Otherwise run base model
	if "p" in mode and amount > 0 and amount <= 1:
		pruner.prune(model)
	if "q" in mode:
		# Create filefolder for temp data of TensorRT
		Path(trt_path).mkdir(parents=True, exist_ok=True)
		# Remove any temp files from previous runs as they can interfere with current run
		filelist = [ f for f in os.listdir(trt_path) ]
		for f in filelist:
			os.remove(os.path.join(trt_path, f))
		# Run ONNX preparation for TensorRT quantization
		prepare(model=(model.to(memory_format=torch.channels_last)), batch_size=batchsize, trt_path=trt_path, dataset=dataset)
	else:
		# If no quantization specified, warm up the model (in case of quantization we would warm up with TensorRT)
		warmup_model(model=model, iterations=10, batch_size=batchsize, C=C, W=W, H=H)
		print("########## Finished Warmup")
	# Measure preparation (pruning / quantization) time
	prep_time = time.perf_counter() - start_time
	if "q" in mode:
		# Run inference with TensorRT
		results = run_tensorrt(inference_type=quantizer, batch_size=batchsize, dataset=dataset, trt_path=trt_path)
	else:
		# Run inference with Torch
		results = run_model(model, dataset, batchsize)
	# Save results
	csv_writer.writerow([model_name, pruner_name, quantizer, dataset_name, str(amount), results["accuracy"], results["inf_time"], orig_model_size, results["model_size"], results["batch_size"], prep_time])
	f_csv.flush()


"""
ITERATION OF PRUNING BENCHMARKS WITH CONFIG
"""
def test_pruning():
	# Iterate over every permutation of configurations
	for b, m, p, d, a in itertools.product(cfg.batchsizes, cfg.models_to_test, cfg.pruners_to_test + cfg.removal_pruners_to_test, cfg.datasets_to_test, cfg.degrees):
		# Get models
		model = cfg.modeldict[m]().model
		# Get datasets
		dataset = cfg.datasetdict[d]()
		# Structured Ln pruning needs further parameters
		if p.startswith("lnstructured"):
			pruner = cfg.prunerdict["lnstructured"](n=cfg.ln_n, dim=cfg.ln_dim, amount=a)
		# Removal pruners initialized differently to zero fill pruners
		elif p in cfg.removal_pruners_to_test:
			pruner = cfg.prunerdict["removalpruner"](amount=a, dataset=dataset, attribution=p, batch_size=b, criterion=torch.nn.CrossEntropyLoss(), device=device)
		else:
			pruner = cfg.prunerdict[p](amount=a)
		try:
			benchmark_pq(model=model, pruner=pruner, dataset=dataset, batchsize=b, mode="p")
		except:
			print("########## Couldn't prune % s" % (m))


"""
ITERATION OF QUANTIZATION BENCHMARKS WITH CONFIG
"""
def test_quant():
	for b, m, q, d in itertools.product(cfg.batchsizes, cfg.models_to_test, cfg.quantizers_to_test, cfg.datasets_to_test):
		model = cfg.modeldict[m]().model
		dataset = cfg.datasetdict[d]()
		try:
			benchmark_pq(model=model, quantizer=q, dataset=dataset, batchsize=b, trt_path=trt_path, mode="q")
		except:
			print("########## Couldn't quantize % s" % (m))


"""
ITERATION OF PRUNING + QUANTIZATION BENCHMARKS WITH CONFIG
"""
def test_pruning_and_quant():
	for b, m, p, q, d, a in itertools.product(cfg.batchsizes, cfg.models_to_test, cfg.pruners_to_test + cfg.removal_pruners_to_test, cfg.quantizers_to_test, cfg.datasets_to_test, cfg.degrees):
		model = cfg.modeldict[m]().model
		dataset = cfg.datasetdict[d]()
		if p.startswith("lnstructured"):
			pruner = prunerdict["lnstructured"](n=cfg.ln_n, dim=cfg.ln_dim, amount=a)
		elif p in cfg.removal_pruners_to_test:
			pruner = cfg.prunerdict["removalpruner"](amount=a, dataset=dataset, attribution=p, batch_size=b, criterion=torch.nn.CrossEntropyLoss(), device=device)
		else:
			pruner = cfg.prunerdict[p](amount=a)
		try:
			benchmark_pq(model=model, pruner=pruner, quantizer=q, dataset=dataset, batchsize=b, trt_path=trt_path, mode="pq")
		except:
			print("########## Couldn't compress % s" % (m))


if __name__ == '__main__':
	for _ in range(iterations):
		# Read config file
		cfg = ConfigReader()
		# Create CSV file
		create_experiment_csv(cfg.exp_path)
		# Prune if pruners specified
		if len(cfg.pruners_to_test + cfg.removal_pruners_to_test) > 0: test_pruning()
		# Quantize if quantizer specified
		if len(cfg.quantizers_to_test) > 0: test_quant()
		# Prune and quantize if both specified
		if len(cfg.pruners_to_test + cfg.removal_pruners_to_test) > 0 and len(cfg.quantizers_to_test) > 0: test_pruning_and_quant()

	f_csv.close()
