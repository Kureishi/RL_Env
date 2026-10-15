from .base import Task
from .cartpole import CartPoleV1, clip_states
from .csv import CsvTask
from .audio import AudioTask, log_mel_features, log_mel_frames
from .gridnav import GridNavV1
from .images import ImageTask
from .media import collect_items, detect_modality, modality_from_index
from .parity import Parity4V1, ParityTask, parity_ceiling
from .sine import SineRegressionV1
from .sequence import SequenceMotifV1, motif_present  # SPEC.md 83 (v0.69)
from .text import TextTask, ngram_features
from .medical import MedicalTabularTask
from .finance import FinanceForecastTask
from .robust import RobustTask
from .signals import SignalTask, signal_features, load_signal  # 92.1 (A1)
from .jsonl import JsonlTask, flatten_record  # 92.2 (A2)
from .midi import MidiTask, midi_features, parse_midi  # 92.3 (A3)
from .pointcloud import (  # 92.4 (A4)
    PointCloudTask,
    load_pointcloud,
    pointcloud_features,
)
from .graph import GraphTask, load_graph  # 92.5 (A5)
from .video import VideoTask, video_features, load_video_frames  # 92.6 (A6)
from .mixed import MixedTask, mixed_row  # 92.7 (A7)

# task registry (SPEC.md 15): new tasks register here, zero core-loop changes
TASKS = {
    "cartpole-v1": CartPoleV1,
    "sine-v1": SineRegressionV1,
    "gridnav-v1": GridNavV1,
    "parity-v1": Parity4V1,  # v0.3: noisy XOR classification (README "Extending")
    # v0.69: native sequence task (SPEC.md 83.1) — offers the conv1d / rnn
    # temporal families (the capabilities={"sequence"} draw, catalog 25.5).
    "sequence-motif-v1": SequenceMotifV1,
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
    # v0.78 (SPEC.md 92, A82): the new modalities — labelled directories
    # (signal / midi / pointcloud / video / mixed) and the JSONL file task
    "signal": SignalTask,      # 92.1 (A1): 1-D signals (.npy/.dat/.sig)
    "jsonl": JsonlTask,        # 92.2 (A2): JSONL / nested-JSON records
    "midi": MidiTask,          # 92.3 (A3): MIDI music (.mid/.midi)
    "pointcloud": PointCloudTask,  # 92.4 (A4): 3-D point clouds (.ply/.obj)
    "graph": GraphTask,        # 92.5 (A5): nodes.csv + edges.csv
    "video": VideoTask,        # 92.6 (A6): video clips (.mp4/.mov/.avi)
    "mixed": MixedTask,        # 92.7 (A7): multi-modality labelled dirs
    # robust-v1 is a wrapper (needs a base task) — not a bare registry entry,
    # so it is exported for direct use / plugins rather than the string map.
}

__all__ = [
    "Task", "CartPoleV1", "clip_states", "CsvTask", "GridNavV1",
    "SineRegressionV1", "Parity4V1", "ParityTask", "parity_ceiling",
    "SequenceMotifV1", "motif_present",  # SPEC.md 83 (v0.69)
    "TASKS",
    "ImageTask", "AudioTask", "log_mel_features", "log_mel_frames",
    "collect_items", "detect_modality", "modality_from_index",
    "TextTask", "ngram_features",
    # v0.47: domain task packs (SPEC.md 61.2, A2)
    "MedicalTabularTask", "FinanceForecastTask", "RobustTask",
    # v0.78: the new modalities (SPEC.md 92, A82)
    "SignalTask", "signal_features", "load_signal",
    "JsonlTask", "flatten_record",
    "MidiTask", "midi_features", "parse_midi",
    "PointCloudTask", "load_pointcloud", "pointcloud_features",
    "GraphTask", "load_graph",
    "VideoTask", "video_features", "load_video_frames",
    "MixedTask", "mixed_row",
]
