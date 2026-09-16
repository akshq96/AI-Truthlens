"""Synthetic face-manipulation techniques used for data augmentation.

Ported from the project's leakage-safe research framework. Each technique takes an
already face-cropped BGR uint8 image and returns a manipulated copy, deterministically
for a given seed. See base.py for the shared interface.
"""
from synthetic.autoencoder_swap import AutoencoderSwapTechnique
from synthetic.blend_warp import BlendWarpTechnique
from synthetic.color_perturb import ColorPerturbTechnique
from synthetic.compression_artifact import CompressionArtifactTechnique
from synthetic.freq_perturb import FreqPerturbTechnique

TECHNIQUES = {
    "blend_warp": BlendWarpTechnique,
    "freq_perturb": FreqPerturbTechnique,
    "compression_artifact": CompressionArtifactTechnique,
    "color_perturb": ColorPerturbTechnique,
    "autoencoder_swap": AutoencoderSwapTechnique,
}


def build(name):
    return TECHNIQUES[name]()
