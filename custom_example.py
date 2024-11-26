from pruners import Pruner
from torch.utils.data import Dataset
from PIL import Image
from os.path import join, isfile
from torchvision import transforms
import torch.nn.utils.prune as prune
import torch.nn as nn
from pq_bench import benchmark_pq
import torch
from os import listdir
import json


imagenet_transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

# Create Dataset object with __init__, __len__ and __getitem__ to satisfy Torch's condition on building a DataLoader out of it
class Imagenet1000(Dataset):
    def __init__(self):
        self.data_path = "data/imagenet_sample"
        self.image_paths = [join(self.data_path, f) for f in listdir(self.data_path,) if isfile(join(self.data_path, f))
                            and f.endswith('.jpg')]
        with open(join(self.data_path, "labels.json")) as json_file:
            json_data = json.load(json_file)
        self.labels = json_data["labels"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = imagenet_transform(image)
        label = self.labels[image_filepath[-10:-4]]

        return image, label

    def __str__(self):
        return "Imagenet1000"


# Define a pretrained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)


# Define a class with a "prune" method and a "name" and "amount" attribute.
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
                print("Pruned " + module.str())


# Call benchmark_pq with defined model, pruner and dataset
benchmark_pq(model=model, pruner=Random(amount=0.1), dataset=Imagenet1000(), mode="p", batchsize=64)
