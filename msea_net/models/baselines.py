from typing import Dict
import torch
import torch.nn as nn
import torchvision.models as models
import timm

from .msea_net import MSEANet
from .heads import StandardHead


class BaselineWrapper(nn.Module):
    """
    Standard wrapper around torchvision/timm backbones to provide uniform output dictionary.
    """
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int = 8):
        super().__init__()
        self.backbone = backbone
        self.classifier = StandardHead(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        output = self.classifier(features)
        output['features'] = features
        return output


def build_baseline_model(model_name: str, num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """
    Factory function for baseline architectures and ablation study variants.
    """
    name = model_name.lower().strip()
    
    # 1. Proposed Model & Ablation Variants
    if name in ['msea_net', 'msea-net', 'proposed']:
        return MSEANet(num_classes=num_classes, pretrained=pretrained)
    
    elif name == 'msea_no_multiscale':
        return MSEANet(num_classes=num_classes, use_multiscale=False, pretrained=pretrained)
        
    elif name == 'msea_no_attention':
        return MSEANet(num_classes=num_classes, use_attention=False, pretrained=pretrained)
        
    elif name == 'msea_no_gff':
        return MSEANet(num_classes=num_classes, use_gff=False, pretrained=pretrained)
        
    elif name == 'msea_no_edl':
        return MSEANet(num_classes=num_classes, use_edl=False, pretrained=pretrained)

    # 2. Standard Baselines
    elif name in ['vgg16', 'vgg-16']:
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        base = models.vgg16(weights=weights)
        in_features = base.classifier[0].in_features
        base.classifier = nn.Identity()
        return BaselineWrapper(base, in_features, num_classes)

    elif name in ['resnet50', 'resnet-50']:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        return BaselineWrapper(base, in_features, num_classes)

    elif name in ['densenet121', 'densenet-121']:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        base = models.densenet121(weights=weights)
        in_features = base.classifier.in_features
        base.classifier = nn.Identity()
        return BaselineWrapper(base, in_features, num_classes)

    elif name in ['inception_v3', 'inception-v3', 'inceptionv3']:
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        base = models.inception_v3(weights=weights, aux_logits=False)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        return BaselineWrapper(base, in_features, num_classes)

    elif name in ['mobilenetv3', 'mobilenet-v3', 'mobilenet_v3']:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_large(weights=weights)
        in_features = base.classifier[0].in_features
        base.classifier = nn.Identity()
        return BaselineWrapper(base, in_features, num_classes)

    elif name in ['efficientnetv2s_plain', 'efficientnetv2_s', 'efficientnet']:
        base = timm.create_model('tf_efficientnetv2_s', pretrained=pretrained, num_classes=0)
        in_features = base.num_features
        return BaselineWrapper(base, in_features, num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models include: "
                         "msea_net, vgg16, resnet50, densenet121, inception_v3, mobilenetv3, efficientnetv2s_plain, "
                         "and ablations: msea_no_multiscale, msea_no_attention, msea_no_gff, msea_no_edl")
