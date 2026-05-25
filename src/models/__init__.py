"""Detector model implementations."""

from .baseline import ZeroShotDetector
from .statistical import StatisticalDetector

__all__ = ["ZeroShotDetector", "StatisticalDetector"]
