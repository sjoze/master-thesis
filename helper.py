import torch
import os
from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)
import onnx
import tensorrt as trt
#import onnx_tensorrt.backend as backend
from torch2trt import torch2trt
from datetime import datetime
import time


timestamp = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

def get_model_size(model):
    name = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
    torch.save(model.state_dict(), name + ".pt")
    size = "%.2f MB" %(os.path.getsize(name + ".pt")/1e6)
    os.remove(name + '.pt')
    return size


def sparse_model(model):

    # Specify the fully-qualified-name of the nn.Linear modules you want to swap
    sparse_config = {
        "seq.0": SemiSparseLinear
    }

    # Swap nn.Linear with SemiSparseLinear, you can run your normal training loop after this step
    swap_linear_with_semi_sparse_linear(model, sparse_config)
    print(get_model_size(model))


def count_zero_weights(model):
    zeros = 0
    total = 0
    for param in model.parameters():
        if param is not None:
            zeros += torch.sum((param == 0).int()).item()
            total += torch.numel(param)
    print("Total number of zero elements in model: " + str(zeros))
    print("Number of zero elements in %: " + str(zeros/total))
    return [zeros, zeros/total]


def convert_to_sparse_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(module.weight.data.to_sparse())
            module.bias = torch.nn.Parameter(module.bias.data.to_sparse())
    print("Model after sparsing: " + get_model_size(model))
    return model


def torch_to_onnx(model, dataloader):
    #onnx_program = torch.onnx.dynamo_export(model, next(iter(dataloader))[0])
    onnx_model = torch.onnx.export(model, next(iter(dataloader))[0], f="model.onnx")


def torch_to_trt(model, dataloader):
    torch_to_onnx(model, dataloader)
    onnx_model = onnx.load("model.onnx")


def torch_to_trt_direct(model, dataloader):
    model_trt = torch2trt(model,  [next(iter(dataloader))[0][0].unsqueeze(0).to(device)])
    return model_trt


def get_engine_size(path):
    engine_file_size = 0
    files = [f for f in os.listdir(path) if '.engine' in f]
    if len(files) > 0:
        engine_file_size = os.path.getsize(path + "/" + files[0])/ (1024 * 1024)
    return engine_file_size











