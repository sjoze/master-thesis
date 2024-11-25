import torch.nn.utils.prune as prune
import torch.nn as nn
import torch
from models import AlexNet, ResNet, DenseNet, GoogLeNet, InceptionV3, MobileNetV2, ShuffleNetV2, SimpleNet, SqueezeNet
from pq_datasets import Imagenette, Imagenet1000, ImageWoof
from torch.utils.data import DataLoader
import numpy as np
from statistics import mean
import time
import tensorflow as tf
from math import floor
from helper import get_model_size, sparse_model, convert_to_sparse_weights
from torchpruner.pruner import Pruner as TorchPruner
from torchsummary import summary
from torchpruner.attributions import (
    WeightNormAttributionMetric,
    RandomAttributionMetric,
    SensitivityAttributionMetric,
    TaylorAttributionMetric,
    APoZAttributionMetric,
    ShapleyAttributionMetric,
    find_best_module_for_attributions
)


prunerdict = {}
def register(cls):
    prunerdict[cls.__name__.lower()] = cls
    return cls

def get_prunerdict():
    return prunerdict


class Pruner:
    def __init__(self, name, amount):
        self.name = name
        self.amount = amount

    def __str__(self):
        return f"{self.name}"

    def prune(self, model):
        pass

    def get_amount(self):
        return self.amount


@register
class Random(Pruner):
    def __init__(self, amount, verbosity=0):
        super().__init__("Random Unstructured", amount)
        self.verbosity = verbosity

    def prune(self, model):

        modules = [module for module in model.modules() if not isinstance(module, nn.Sequential) and hasattr(module, "weight")]

        for module in modules:
            prune.random_unstructured(module, name="weight", amount=self.amount)
            prune.remove(module, 'weight')
            if self.verbosity > 0:
                print("Pruned " + module.__str__())


# default: n = -inf, dim = 1
# Disconnect all connections to one input: 1 (or Channels for Convs)
# Disconnect one neuron: 0 (or Neurons for Convs)
@register
class LnStructured(Pruner):
    def __init__(self, amount, dim=1, n=float('-inf'), verbosity=0):
        super().__init__("Ln Structured", amount)
        self.dim = dim
        self.n = n
        self.verbosity = verbosity

    def prune(self, model):

        modules = [module for module in model.modules() if not isinstance(module, nn.Sequential) and hasattr(module, "weight")]

        for module in modules:
            #try:
            prune.ln_structured(module, name="weight", amount=self.amount, dim=self.dim, n=self.n)
            prune.remove(module, 'weight')
            #except:
            #    continue
            if self.verbosity > 0:
                print("Pruned " + module.__str__())


@register
class L1Unstructured(Pruner):
    def __init__(self, amount, verbosity=0):
        super().__init__("L1 Unstructured", amount)
        self.verbosity = verbosity

    def prune(self, model):

        modules = [module for module in model.modules() if not isinstance(module, nn.Sequential) and hasattr(module, "weight")]

        for module in modules:
            prune.l1_unstructured(module, name="weight", amount=self.amount)
            prune.remove(module, 'weight')
            if self.verbosity > 0:
                print("Pruned " + module.__str__())


@register
class L1UnstructuredGlobal(Pruner):
    def __init__(self, amount, verbosity=0):
        super().__init__("L1 Unstructured Global", amount)
        self.verbosity = verbosity

    def prune(self, model):
        modules = [module for module in model.modules() if
                   not isinstance(module, nn.Sequential) and hasattr(module, "weight")]
        parameters = []
        for module in modules:
            parameters.append((module, "weight"))

        prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=self.amount)
        for parameter in parameters:
            prune.remove(parameter[0], parameter[1])
        for module in modules:
            if self.verbosity > 0:
                print("Pruned " + module.__str__())

@register
class RemovalPruner(Pruner):
    def __init__(self, amount, dataset, attribution, batch_size, criterion, device):
        super().__init__("Removal Pruner", amount)
        self.attribution_dict = {
            "random": RandomAttributionMetric, "weight": WeightNormAttributionMetric, "sensitivity": SensitivityAttributionMetric,
            "taylor": TaylorAttributionMetric }
        self.attribution = attribution
        self.batch_size = batch_size
        self.criterion = criterion
        self.device = device
        self.dataset = dataset

    def get_pruning_graph(self, model):
            already_added = []
            modules = [module for module in model.modules()]
            pruning = []
            current = None

            for module in modules:
                if any([isinstance(module, c) for c in [nn.Linear, nn.Conv2d]]):
                    if current is not None:
                        pruning[-1][1].append(module)
                        pruning[-1][1].reverse()
                    current = module
                    pruning.append((module, []))
                elif (
                    any([isinstance(module, c) for c in [nn.BatchNorm2d, nn.Dropout]])
                    and current is not None
                ):
                    pruning[-1][1].append(module)
                    already_added.append(module)
            return pruning

    def prune(self, model):
        pruning_graph = self.get_pruning_graph(model)
        C, W, H = self.dataset.__getitem__(0)[0].shape
        data_generator = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        attr = self.attribution_dict[self.attribution](model, data_generator, self.criterion, self.device)
        pruner = TorchPruner(model, input_size=(C, W, H), device=self.device)

        for module, cascading_modules in pruning_graph[::-1]:		#[::-1]
            if len(cascading_modules) == 0:
                continue
            if isinstance(module, torch.nn.modules.linear.Linear) or isinstance(module, torch.nn.modules.conv._ConvNd):
                scores = attr.run(module)
                scores_copy = np.copy(scores)
                scores_copy.sort()
                threshold = scores_copy[floor(scores.size * self.amount)]
                pruning_indices = np.argwhere(scores < threshold).flatten()
                pruner.prune_model(module, pruning_indices, cascading_modules=cascading_modules)

    def __str__(self):
        return f"{self.attribution}"








