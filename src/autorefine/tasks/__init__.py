from .base import Task
from .cartpole import CartPoleV1, clip_states
from .csv import CsvTask
from .gridnav import GridNavV1
from .parity import Parity4V1, ParityTask, parity_ceiling
from .sine import SineRegressionV1

# task registry (SPEC.md 15): new tasks register here, zero core-loop changes
TASKS = {
    "cartpole-v1": CartPoleV1,
    "sine-v1": SineRegressionV1,
    "gridnav-v1": GridNavV1,
    "parity-v1": Parity4V1,  # v0.3: noisy XOR classification (README "Extending")
    "csv": CsvTask,  # v0.8: user CSV data (SPEC.md 22.1; needs task_config.path)
}

__all__ = [
    "Task", "CartPoleV1", "clip_states", "CsvTask", "GridNavV1",
    "SineRegressionV1", "Parity4V1", "ParityTask", "parity_ceiling", "TASKS",
]
