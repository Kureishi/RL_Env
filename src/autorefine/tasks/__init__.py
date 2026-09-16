from .base import Task
from .cartpole import CartPoleV1, clip_states
from .csv import CsvTask
from .audio import AudioTask, log_mel_features, log_mel_frames
from .gridnav import GridNavV1
from .images import ImageTask
from .media import collect_items, detect_modality, modality_from_index
from .parity import Parity4V1, ParityTask, parity_ceiling
from .sine import SineRegressionV1
from .text import TextTask, ngram_features
from .medical import MedicalTabularTask
from .finance import FinanceForecastTask
from .robust import RobustTask

# task registry (SPEC.md 15): new tasks register here, zero core-loop changes
TASKS = {
    "cartpole-v1": CartPoleV1,
    "sine-v1": SineRegressionV1,
    "gridnav-v1": GridNavV1,
    "parity-v1": Parity4V1,  # v0.3: noisy XOR classification (README "Extending")
    "csv": CsvTask,  # v0.8: user CSV data (SPEC.md 22.1; needs task_config.path)
    # v0.10: input modalities (SPEC.md 24.3/24.4; need task_config.path dir)
    "image": ImageTask,
    "audio": AudioTask,
    # v0.31: text modality (SPEC.md 45.2; needs task_config.path dir)
    "text": TextTask,
    # v0.47: domain task packs (SPEC.md 61.2, A2; additive — the §21 loader
    # delivers the same registry an external entry-point would use)
    "medical-v1": MedicalTabularTask,  # imbalanced cost-sensitive binary
    "finance-v1": FinanceForecastTask,  # temporal horizon forecasting
    # robust-v1 is a wrapper (needs a base task) — not a bare registry entry,
    # so it is exported for direct use / plugins rather than the string map.
}

__all__ = [
    "Task", "CartPoleV1", "clip_states", "CsvTask", "GridNavV1",
    "SineRegressionV1", "Parity4V1", "ParityTask", "parity_ceiling", "TASKS",
    "ImageTask", "AudioTask", "log_mel_features", "log_mel_frames",
    "collect_items", "detect_modality", "modality_from_index",
    "TextTask", "ngram_features",
    # v0.47: domain task packs (SPEC.md 61.2, A2)
    "MedicalTabularTask", "FinanceForecastTask", "RobustTask",
]
