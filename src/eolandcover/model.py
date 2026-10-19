"""Models and embeddings for EuroSAT (optional ``torch`` extra).

Planned (roadmap), in increasing ambition:
  * a pretrained CNN fine-tuned (or used as a fixed feature extractor) for the
    10-class land-cover task;
  * the same CNN with its classification head removed, used to produce
    *embeddings* that feed `eolandcover.change` for change detection.

Not yet implemented — install the extra when building this out:
    pip install -e '.[torch]'
"""
from __future__ import annotations


def build_classifier(num_classes: int = 10, in_channels: int = 13):
    """Return a CNN classifier for EuroSAT. *(Planned — not yet implemented.)*

    Intended approach: use a TorchGeo model pretrained on satellite imagery
    (e.g. ``from torchgeo.models import resnet18, ResNet18_Weights``), built for
    ``in_channels`` (Sentinel-2 has 13 bands) and ``num_classes`` outputs — first
    as a frozen feature extractor, then optionally fine-tuned.
    """
    raise NotImplementedError("build_classifier is planned — see the roadmap in the README.")


def embed(model, images):
    """Return embeddings (features before the classification head). *(Planned.)*

    Intended approach: run ``images`` through ``model`` with the final linear layer
    replaced by an identity, returning the vectors as a NumPy array suitable for
    ``eolandcover.change.change_scores``.
    """
    raise NotImplementedError("embed is planned — see the roadmap in the README.")
