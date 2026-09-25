"""
Calibrators package.

All calibrators use logits-based API:
    calibrator = Calibrator(num_classes, device, ...)
    calibrator.fit(logits, labels, val_idx, train_idx, ...)
    calibrated_logits = calibrator.calibrate(logits)
"""

from .ts import TS, fit_calibration
from .vs import VS
from .ets import ETS
from .cagcn import CaGCN
from .gats import GATS
from .gets import GETS
from .hts import HTS
from .wats import WATS
from .dcgc import DCGC
from .hots import HoTS

__all__ = [
    'TS', 'VS', 'ETS', 'CaGCN', 'GATS', 'GETS', 'HTS', 'WATS', 'DCGC',
    'HoTS', 'fit_calibration',
]
