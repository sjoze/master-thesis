import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights


# Return all available models
modeldict = {}
def register(cls):
    modeldict[cls.__name__.lower()] = cls
    return cls

def get_modeldict():
    return modeldict

class Model:
    def __init__(self):
        pass

    def __str__(self):
        pass


@register
class AlexNet(Model):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=True)



@register
class DenseNet(Model):
    def __init__(self, layers=121):
        super().__init__()
        self.layers = layers
        try:
            self.model = torch.hub.load("pytorch/vision:v0.10.0", "densenet" + str(self.layers), pretrained=True)
        except:
            raise Exception(str(self.layers) +
                            " is not a supported amount of layers. DenseNet supports: 121, 161, 169, and 201.")

    def __str__(self):
        return f"DenseNet with {self.layers} layers"


@register
class GoogLeNet(Model):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "googlenet", pretrained=True)


@register
class ResNet(Model):
    def __init__(self, layers=18):
        super().__init__()
        self.layers = layers
        try:
            self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet"+str(self.layers), pretrained=True)
        except:
            raise Exception(str(self.layers) +
                            " is not a supported amount of layers. ResNet supports: 18, 34, 50, 101 and 152.")

    def __str__(self):
        return f"ResNet with {self.layers} layers"


@register
class InceptionV3(Model):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)


@register
class MobileNetV2(Model):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)


@register
class ShuffleNetV2(Model):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "shufflenet_v2_x1_0", pretrained=True)


@register
class SimpleNet(Model):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_5m_m1", pretrained=True)
        # other variants: simplenetv1_5m_m2, simplenetv1_9m_m1, simplenetv1_9m_m2, simplenetv1_small_m1_05,
        # simplenetv1_small_m2_05, simplenetv1_small_m1_075, simplenetv1_small_m2_075


@register
class SqueezeNet(Model):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_0", pretrained=True)
        # other variants: squeezenet1_1  


@register
class VisionTransformer(Model):
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights = ViT_B_16_Weights)
