import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights


modeldict = {}
def register(cls):
    modeldict[cls.__name__.lower()] = cls
    return cls

def get_modeldict():
    return modeldict

class Model:
    def __init__(self, name):
        self.name = name

    def pretrained(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def __str__(self):
        return f"{self.name}"


@register
class AlexNet(Model):
    def __init__(self, pretrained=True):
        super().__init__("AlexNet")
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=pretrained) 



@register
class DenseNet(Model):
    def __init__(self, layers=121, pretrained=True):
        super().__init__("DenseNet")
        self.layers = layers
        try:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet' + str(self.layers), pretrained=pretrained)
        except:
            raise Exception(str(self.layers) +
                            " is not a supported amount of layers. DenseNet supports: 121, 161, 169, and 201.")

    def __str__(self):
        return f"{self.name} with {self.layers} layers"


@register
class GoogLeNet(Model):
    def __init__(self, pretrained=True):
        super().__init__("GoogLeNet")
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=pretrained)  


@register
class ResNet(Model):
    def __init__(self, layers=18, pretrained=True):
        super().__init__("ResNet")
        self.layers = layers
        try:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet'+str(self.layers), pretrained=pretrained)
        except:
            raise Exception(str(self.layers) +
                            " is not a supported amount of layers. ResNet supports: 18, 34, 50, 101 and 152.")

    def __str__(self):
        return f"{self.name} with {self.layers} layers"  


@register
class InceptionV3(Model):
    def __init__(self, pretrained=True):
        super().__init__("InceptionV3")
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=pretrained) 


@register
class MobileNetV2(Model):
    def __init__(self, pretrained=True):
        super().__init__("MobileNetV2")
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)  


@register
class ShuffleNetV2(Model):
    def __init__(self, pretrained=True):
        super().__init__("ShuffleNetV2")
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=pretrained)  


@register
class SimpleNet(Model):
    def __init__(self, pretrained=True):
        super().__init__("SimpleNet")
        self.model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_5m_m1", pretrained=pretrained)
        # other variants: simplenetv1_5m_m2, simplenetv1_9m_m1, simplenetv1_9m_m2, simplenetv1_small_m1_05,
        # simplenetv1_small_m2_05, simplenetv1_small_m1_075, simplenetv1_small_m2_075


@register
class SqueezeNet(Model):
    def __init__(self, pretrained=True):
        super().__init__("SqueezeNet")
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_0", pretrained=pretrained)
        # other variants: squeezenet1_1  


@register
class VisionTransformer(Model):
    def __init__(self, pretrained=True):
        super().__init__("SqueezeNet")
        self.model = vit_b_16(weights = ViT_B_16_Weights)
        # other variants: squeezenet1_1

