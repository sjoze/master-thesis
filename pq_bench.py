from configparser import ConfigParser
import ast
from models import get_modeldict
from pq_datasets import get_datasetdict
from pruners import get_prunerdict
#import numpy as np
import time
from statistics import mean
from torch.utils.data import DataLoader
import tensorflow as tf
import torch
import csv
from helper import get_model_size
from datetime import datetime
from quantize import prepare, run_tensorrt
from torch.profiler import profile, record_function, ProfilerActivity
import os
#from torchsummary import summary
import itertools


device = torch.device("cuda")
#device = torch.device("cpu")
print(f"########## CUDNN IS AVAILABLE: {torch.backends.cudnn.is_available()}")
print("########## WORKING ON: " + torch.cuda.get_device_name())

# Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")

# Get the sections
pruning = config_object["PRUNING"]
quantizing = config_object["QUANTIZING"]
models = config_object["MODELS"]
datasets = config_object["DATASETS"]
pruners = config_object["PRUNERS"]
parameters = config_object["PARAMETERS"]
removal = config_object["REMOVAL_PRUNERS"]
ln_dim = int(config_object['LN_PRUNING']['dim'])
ln_n = int(config_object['LN_PRUNING']['n'])
trt_path = config_object['TRT_FILEFOLDER']['path']
degrees = ast.literal_eval(pruning["degrees"])
batchsizes = ast.literal_eval(parameters["batchsizes"])
iterations = int(config_object['PARAMETERS']['iterations'])

# Filter out models, datasets, pruners and quantizers
modeldict = get_modeldict()
models_to_test = [m for m in models if models.getboolean(m)]

datasetdict = get_datasetdict()
datasets_to_test = [d for d in datasets if datasets.getboolean(d)]

prunerdict = get_prunerdict()
pruners_to_test = [p for p in pruners if pruners.getboolean(p)]
removal_pruners_to_test = [p for p in removal if removal.getboolean(p)]

quantizers_to_test = [q for q in quantizing if quantizing.getboolean(q)]

# Create csv file for results
timestamp = datetime.now().strftime("%m:%d:%Y_%H:%M:%S")
csv_fields = ["Model", "Pruner", "Quantizer", "Dataset", "Amount", "Accuracy", "Inference Time (ms)", "Original Size", "Compressed Size", "Batch Size", "Prep Time (s)"]
f_csv = open("exps/Experiment_" + timestamp + ".csv", "a")
csv_writer = csv.writer(f_csv)
csv_writer.writerow(csv_fields)


"""
MODEL EXECUTION
"""
def run_model(model, dataset, batch_size):
	data_generator = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
	accs = []
	inference_times = []
	outputs_to_keep = dataset.outputs_to_keep()
	for x, y in data_generator:
		x, y = x.to(device), y
		with profile(activities=[
			ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
			with record_function("model_inference"):
				output = model(x)
		for event in prof.key_averages():
			if event.key == "model_inference":
				inference_times.append(event.cuda_time)

		output = output.cpu().detach()
		outputs_to_keep = dataset.outputs_to_keep()
		if len(outputs_to_keep) > 0:
			output = torch.from_numpy(tf.transpose(tf.gather(tf.transpose(output), outputs_to_keep)).numpy())
		output = torch.argmax(output, dim=1)
		acc = (output.round() == y).float().mean()
		acc = float(acc * 100)
		accs.append(float(acc))
	print("########## Accuracy: " + "%.2f%%" % (mean(accs)))
	print(f"########## Model elapsed: {mean(inference_times)} ms")
	return {"accuracy": "%.2f" % (mean(accs)), "inf_time": mean(inference_times), "model_size": get_model_size(model), "batch_size": batch_size}


def warmup_model(model, iterations, batch_size, C, H, W):
	for i in range(iterations):
		dummy_data = torch.rand(batch_size, C, H, W).to(device)
		_ = model(dummy_data)


"""
MODEL COMPRESSION AND BENCHMARKING
"""
def benchmark_pq(model, dataset, batchsize, mode, pruner="-", quantizer="-", trt_path="./data/trt_caches"):

	model = model.eval().to(device)
	amount = pruner_name = "-"
	if "p" in mode:
		amount = pruner.get_amount()
		pruner_name = pruner.__str__()

	model_name, dataset_name = [model.__class__.__name__, dataset.__str__()]
	orig_model_size = get_model_size(model)
	C, W, H = dataset.__getitem__(0)[0].shape

	print("########## MODEL: % s with PRUNER: % s and QUANTIZER: % s and AMOUNT: % s on DATASET: %s with BATCHSIZE: %s" % (model_name, pruner_name, quantizer, amount, dataset_name, batchsize))
	start_time = time.perf_counter()
	if "p" in mode and amount > 0 and amount <= 1:
		pruner.prune(model)
	if "q" in mode:
		os.makedirs(trt_path, exist_ok=True)
		filelist = [ f for f in os.listdir(trt_path) ]
		for f in filelist:
			os.remove(os.path.join(trt_path, f))
		prepare(model=(model.to(memory_format=torch.channels_last)), batch_size=batchsize, trt_path=trt_path, dataset=dataset)
	else:
		warmup_model(model=model, iterations=10, batch_size=batchsize, C=C, W=W, H=H)
		print("########## Finished Warmup")
	prep_time = time.perf_counter() - start_time
	if "q" in mode:
		results = run_tensorrt(inference_type=quantizer, batch_size=batchsize, dataset=dataset, trt_path=trt_path)
	else:
		results = run_model(model, dataset, batchsize)
	csv_writer.writerow([model_name, pruner_name, quantizer, dataset_name, str(amount), results["accuracy"], results["inf_time"], orig_model_size, results["model_size"], results["batch_size"], prep_time])
	f_csv.flush()


"""
ITERATION OF PRUNING BENCHMARKS WITH CONFIG
"""
def test_pruning():
	for b, m, p, d, a in itertools.product(batchsizes, models_to_test, pruners_to_test + removal_pruners_to_test, datasets_to_test, degrees):
		model = modeldict[m]().model
		dataset = datasetdict[d]()
		if p.startswith("lnstructured"):
			pruner = prunerdict["lnstructured"](n=ln_n, dim=ln_dim, amount=a)
		elif p in removal_pruners_to_test:
			pruner = prunerdict["removalpruner"](amount=a, dataset=dataset, attribution=p, batch_size=b, criterion=torch.nn.CrossEntropyLoss(), device=device)
		else:
			pruner = prunerdict[p](amount=a)
		try:
			benchmark_pq(model=model, pruner=pruner, dataset=dataset, batchsize=b, mode="p")
		except:
			print("########## Couldn't prune % s" % (m))


"""
ITERATION OF QUANTIZATION BENCHMARKS WITH CONFIG
"""
def test_quant():
	for b, m, q, d in itertools.product(batchsizes, models_to_test, quantizers_to_test, datasets_to_test):
		model = modeldict[m]().model
		dataset = datasetdict[d]()
		#try:
		benchmark_pq(model=model, quantizer=q, dataset=dataset, batchsize=b, trt_path=trt_path, mode="q")
		#except:
		#	print("########## Couldn't quantize % s" % (m))


"""
ITERATION OF PRUNING + QUANTIZATION BENCHMARKS WITH CONFIG
"""
def test_pruning_and_quant():
	for b, m, p, q, d, a in itertools.product(batchsizes, models_to_test, pruners_to_test + removal_pruners_to_test, quantizers_to_test, datasets_to_test, degrees):
		model = modeldict[m]().model
		dataset = datasetdict[d]()
		if p.startswith("lnstructured"):
			pruner = prunerdict["lnstructured"](n=ln_n, dim=ln_dim, amount=a)
		elif p in removal_pruners_to_test:
			pruner = prunerdict["removalpruner"](amount=a, dataset=dataset, attribution=p, batch_size=b, criterion=torch.nn.CrossEntropyLoss(), device=device)
		else:
			pruner = prunerdict[p](amount=a)
		try:
			benchmark_pq(model=model, pruner=pruner, quantizer=q, dataset=dataset, batchsize=b, trt_path=trt_path, mode="pq")
		except:
			print("########## Couldn't compress % s" % (m))


for _ in range(iterations):
	if len(pruners_to_test + removal_pruners_to_test) > 0: test_pruning()
	if len(quantizers_to_test) > 0: test_quant()
	if len(pruners_to_test + removal_pruners_to_test) > 0 and len(quantizers_to_test) > 0: test_pruning_and_quant()

f_csv.close()
