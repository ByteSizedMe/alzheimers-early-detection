"""Early detection of Alzheimer's disease from MRI and clinical records."""

__version__ = "0.1.0"

CLASSES = ("NC", "EMCI", "LMCI", "AD")

CLASS_DESCRIPTIONS = {
    "NC": "Normal cognition",
    "EMCI": "Early mild cognitive impairment",
    "LMCI": "Late mild cognitive impairment",
    "AD": "Alzheimer's disease",
}
