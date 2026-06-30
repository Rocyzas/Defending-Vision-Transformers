import timm
import torch.nn as nn

def build_vit_base_patch16_224(num_classes: int, img_size: int = 224) -> nn.Module:
    model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=img_size)
    n_inputs = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(n_inputs, num_classes),
    )
    return model


def build_vit_base_r50_s16_224(num_classes: int, img_size: int = 224) -> nn.Module:
    model = timm.create_model('vit_base_r50_s16_224.orig_in21k', pretrained=True, img_size=img_size)
    n_inputs = model.num_features
    model.head = nn.Sequential(
        nn.Linear(n_inputs, num_classes),
    )
    return model
