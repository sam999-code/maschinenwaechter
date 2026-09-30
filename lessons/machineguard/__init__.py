"""MachineGuard: our consolidated, tested product package.

One simulator. One model. One pipeline. No copy-paste.
Everything the lessons taught us, in its final home.
"""

from .simulator import simulate_machine
from .model import LSTMAutoencoder
from .pipeline import (make_windows, fit_scaler, apply_scaler,
                       train_autoencoder, score_windows,
                       calibrate_threshold, evaluate, sustained_alerts)

__all__ = ["simulate_machine", "LSTMAutoencoder", "make_windows",
           "fit_scaler", "apply_scaler", "train_autoencoder",
           "score_windows", "calibrate_threshold", "evaluate",
           "sustained_alerts"]
