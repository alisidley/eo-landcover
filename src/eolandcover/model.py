"""Models and embeddings for EuroSAT (optional ``torch`` extra).

Two routes, in increasing ambition:
  * a pretrained CNN fine-tuned (or used as a fixed feature extractor) for the
    10-class land-cover task;
  * the same CNN with its classification head removed, used to produce
    *embeddings* that feed `eolandcover.change` for change detection.

Guided scaffold — install the extra first:  pip install -e '.[torch]'
"""
from __future__ import annotations


def build_classifier(num_classes: int = 10, in_channels: int = 13):
    """Return a CNN classifier for EuroSAT.

    TODO (your turn):
      1. TorchGeo ships models pretrained on satellite imagery, e.g.
         `from torchgeo.models import resnet18, ResNet18_Weights`.
      2. Build a ResNet-18 with `in_channels` (Sentinel-2 has 13 bands) and
         `num_classes` outputs. Start with weights frozen (feature extractor),
         then try fine-tuning.
      3. Return the model.
    """
    raise NotImplementedError("Build a (pretrained) classifier — see the TODO.")


def embed(model, images):
    """Return embeddings (features before the classification head) for a batch.

    TODO (your turn):
      1. Take `model` from `build_classifier` and run images through everything
         except the final linear layer (e.g. replace the head with `nn.Identity`).
      2. Return the resulting vectors as a NumPy array.
      3. Feed before/after embeddings to `eolandcover.change.change_scores`.
    """
    raise NotImplementedError("Produce embeddings — see the TODO.")
