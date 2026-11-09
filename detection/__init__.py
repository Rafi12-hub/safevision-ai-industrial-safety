"""Detection package for Industrial Safety Monitoring System."""
from .detector import SafetyDetector
from .tracker import WorkerTracker
from .safety_checker import SafetyVerifier, annotate_frame

