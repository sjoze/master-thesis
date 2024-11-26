from torch.utils.data import Dataset
import json
from PIL import Image
from os.path import join, isfile
from os import listdir
from torchvision import transforms
from torchvision import datasets
import os
from pathlib import Path

# Define general transform operations for Imagenet and Imagenet-like datasets
imagenet_transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

datasetdict = {}
def register(cls):
    datasetdict[cls.__name__.lower()] = cls
    return cls

def get_datasetdict():
    return datasetdict


@register
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

@register
class ImageWoof(Dataset):
    # 193 = Australian terrier, 182 = Border terrier, 258 = Samoyed, 162 = Beagle, 155 = Shih-Tzu,
    # 167 = English foxhound, 159 = Rhodesian ridgeback, 273 = Dingo, 207 = Golden retriever, 229 = Old English sheepdog
    def __init__(self):
        self.data_path = "data/imagewoof2/val"
        self.image_paths = []
        self.labels = []
        self.outputs_to_keep = [155, 159, 162, 167, 193, 182, 207, 229, 258, 273]
        self.label_dict = dict(
              n02086240=0,
              n02087394=1,
              n02088364=2,
              n02089973=3,
              n02093754=4,
              n02096294=5,
              n02099601=6,
              n02105641=7,
              n02111889=8,
              n02115641=9
            )
        for path, subdirs, files in os.walk(self.data_path):
            for name in files:
                self.image_paths.append(os.path.join(path, name))
                self.labels.append(self.label_dict[os.path.basename(path)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = imagenet_transform(image.convert('RGB'))        # Some images are greyscale
        label = self.labels[idx]

        return image, label

    def __str__(self):
        return "ImageWoof"


@register
class Imagenette(Dataset):
    # 0 = tench, 217 = English springer, 482 = cassette player, 491 = chain saw, 497 = church,
    # 566 = French horn, 569 = garbage truck, 571 = gas pump, 574 = golf ball, 701 = parachute
    def __init__(self, split="val"):
        self.dataset = datasets.Imagenette(split=split, root="data", transform=imagenet_transform)
        self.outputs_to_keep = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __str__(self):
        return "ImageNette"

