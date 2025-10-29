import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

###############################################################################
# LPIPS (Learned Perceptual Image Patch Similarity)
# Inspired by https://github.com/richzhang/PerceptualSimilarity
###############################################################################

class LPIPS(nn.Module):
    def __init__(self, net='vgg'):
        """
        LPIPS metric using pretrained feature extractors.

        Args:
            net (str): Backbone network type ('vgg' | 'alex'). Default: 'vgg'.
        """
        super(LPIPS, self).__init__()

        if net == 'vgg':
            pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            self.layers = nn.Sequential(*list(pretrained)[:23])  # conv1_2 to conv4_3
            self.feature_layers = [3, 8, 15, 22]  # conv1_2, conv2_2, conv3_3, conv4_3
        elif net == 'alex':
            pretrained = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features
            self.layers = pretrained
            self.feature_layers = [2, 5, 8, 10]
        else:
            raise ValueError("Unsupported net type: choose from ['vgg', 'alex']")

        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False

        # Precomputed normalization stats for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward_once(self, x):
        """Extract multi-layer features from the input image."""
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.feature_layers:
                feats.append(x)
        return feats

    def forward(self, x, y):
        """
        Compute LPIPS distance between two images x and y.
        Both should be in [-1, 1] range and shape [B, 3, H, W].
        """
        # Convert [-1, 1] → [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2

        # Normalize to match ImageNet statistics
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        feats_x = self.forward_once(x)
        feats_y = self.forward_once(y)

        loss = 0
        for fx, fy in zip(feats_x, feats_y):
            fx = F.normalize(fx, dim=1)
            fy = F.normalize(fy, dim=1)
            loss += ((fx - fy) ** 2).mean([1, 2, 3])  # mean over channels & spatial dims

        return loss.mean()  # mean over batch
