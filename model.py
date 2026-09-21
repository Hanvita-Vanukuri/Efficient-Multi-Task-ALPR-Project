import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from dataset import CHARS, NUM_CLASSES


class LightEdgeALPR(nn.Module):
    """
    Light-Edge style ALPR recognition model.

    Backbone:
        ResNet-18

    Feature extraction:
        FPN-style multi-scale feature fusion

    Recognition:
        CNN features -> sequence -> Linear classifier -> CTC
    """

    def __init__(self, num_classes=None):
        super().__init__()

        # Always use the vocabulary defined in dataset.py
        self.num_classes = NUM_CLASSES

        if num_classes is not None and num_classes != NUM_CLASSES:
            raise ValueError(
                f"Model class mismatch. "
                f"dataset.py expects {NUM_CLASSES} classes, "
                f"but received {num_classes}."
            )

        # ResNet-18 backbone
        backbone = resnet18(weights=None)

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1   # 64 channels
        self.layer2 = backbone.layer2   # 128 channels
        self.layer3 = backbone.layer3   # 256 channels
        self.layer4 = backbone.layer4   # 512 channels

        # FPN lateral layers
        self.lat2 = nn.Conv2d(128, 128, kernel_size=1)
        self.lat3 = nn.Conv2d(256, 128, kernel_size=1)
        self.lat4 = nn.Conv2d(512, 128, kernel_size=1)

        # 1x1 channel-fusion layer
        self.channel_fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Recognition classifier
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        # -------------------------
        # ResNet-18 backbone
        # -------------------------
        x = self.layer0(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # -------------------------
        # FPN-style feature fusion
        # -------------------------

        p4 = self.lat4(c5)

        p3_base = self.lat3(c4)

        p4_up = F.interpolate(
            p4,
            size=p3_base.shape[-2:],
            mode="nearest"
        )

        p3 = p3_base + p4_up

        p2_base = self.lat2(c3)

        p3_up = F.interpolate(
            p3,
            size=p2_base.shape[-2:],
            mode="nearest"
        )

        p2 = p2_base + p3_up

        # -------------------------
        # 1x1 channel fusion
        # -------------------------
        features = self.channel_fusion(p2)

        # -------------------------
        # Convert image features
        # to character sequence
        # -------------------------
        #
        # features shape:
        # [B, 128, H, W]
        #
        # Mean over height:
        # [B, 128, W]
        #
        # Permute:
        # [B, W, 128]
        #
        features = features.mean(dim=2)
        features = features.permute(0, 2, 1)

        # -------------------------
        # Character classification
        # -------------------------
        logits = self.classifier(features)

        # logits shape:
        # [Batch, Time, Classes]
        #
        # This is exactly what the
        # training and prediction code expects.
        return {
            "recognition": logits
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing LightEdgeALPR")
    print("=" * 60)

    print("Characters :", len(CHARS))
    print("CTC classes:", NUM_CLASSES)

    model = LightEdgeALPR()

    dummy = torch.randn(2, 3, 32, 160)

    with torch.no_grad():
        output = model(dummy)

    recognition = output["recognition"]

    print("Input shape :", tuple(dummy.shape))
    print("Output shape:", tuple(recognition.shape))
    print("Expected    : [Batch, Time, Classes]")

    print("=" * 60)
    print("Model test completed successfully.")
    print("=" * 60)