import torch
import os
from torch2trt import torch2trt
from datetime import datetime
from configparser import ConfigParser
import ast
from models import get_modeldict
from pq_datasets import get_datasetdict
from pruners import get_prunerdict
import tensorflow as tf


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# Class for reading in config sections
class ConfigReader():
    def __init__(self):
        config_object = ConfigParser()
        config_object.read("config.ini")
        self.pruning = config_object["PRUNING"]
        self.quantizing = config_object["QUANTIZING"]
        self.models = config_object["MODELS"]
        self.datasets = config_object["DATASETS"]
        self.pruners = config_object["PRUNERS"]
        self.parameters = config_object["PARAMETERS"]
        self.removal = config_object["REMOVAL_PRUNERS"]
        self.ln_dim = int(config_object["LN_PRUNING"]["dim"])
        self.ln_n = int(config_object["LN_PRUNING"]["n"])
        self.trt_path = config_object["FILEFOLDERS"]["trt_caches"]
        self.exp_path = config_object["FILEFOLDERS"]["experiments"]
        self.degrees = ast.literal_eval(self.pruning["degrees"])
        self.batchsizes = ast.literal_eval(self.parameters["batchsizes"])
        self.iterations = int(config_object["PARAMETERS"]["iterations"])

        self.modeldict = get_modeldict()
        self.models_to_test = [m for m in self.models if self.models.getboolean(m)]

        self.datasetdict = get_datasetdict()
        self.datasets_to_test = [d for d in self.datasets if self.datasets.getboolean(d)]

        self.prunerdict = get_prunerdict()
        self.pruners_to_test = [p for p in self.pruners if self.pruners.getboolean(p)]
        self.removal_pruners_to_test = [p for p in self.removal if self.removal.getboolean(p)]

        self.quantizers_to_test = [q for q in self.quantizing if self.quantizing.getboolean(q)]


def get_model_size(model):
	"""
	Return size of Torch model in MB

	Args:
		model (Torch Model): 	Model to be measured

	Returns:
		String:					Size in MB as string
	"""
    name = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
    torch.save(model.state_dict(), name + ".pt")
    size = "%.2f MB" %(os.path.getsize(name + ".pt")/1e6)
    os.remove(name + ".pt")
    return size


def count_zero_weights(model):
	"""
	Checks if zero fill pruner pruned accordingly

	Args:
		model (Torch Model): 	Model to be checked

	Returns:
		List:					Returns the total number of zero entries and the ratio in the model
	"""
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
	"""
	Sparsifies a model and prints out new model size

	Args:
		model (Torch Model): 	Model to be converted

	Returns:
		Torch Model:			Sparsified model
	"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(module.weight.data.to_sparse())
            module.bias = torch.nn.Parameter(module.bias.data.to_sparse())
    print("Model after sparsing: " + get_model_size(model))
    return model


def calculate_accuracy(dataset, output, y, mode):
	"""
	Returns accuracy value from classification of specified classes

	Args:
		dataset (Torch Dataset): 	Dataset used in the testing process for calulcating predictions
        output (NumPy Array):       Output of the testing process
        y (Tensort):                Labels of the tested data
        mode (String):              Indicator whether pruning or quantization has been used

	Returns:
		Float:			            Accuracy of output
	"""
    # If dataset is on subset of data, only take those classes into account
    if hasattr(dataset, "outputs_to_keep"):
        output = torch.from_numpy(tf.transpose(tf.gather(tf.transpose(output), dataset.outputs_to_keep)).numpy())
    # Quantization output needs to be reshaped
    if "q" in mode:
        output = torch.squeeze(output, dim=0)
    output = torch.argmax(output, dim=1)
    acc = (output.round() == y).float().mean()
    return float(acc * 100)

