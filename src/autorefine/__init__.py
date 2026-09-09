"""AutoRefine: environment for autonomous iterative ML model improvement."""
from .dashboard import DashboardRunner
from .config import (
    ACTIVATIONS,
    BATCH_SIZES,
    DEFAULT_SPEC,
    HIDDEN_LAYER_SIZES,
    LR_SCHEDULES,
    MODEL_FAMILIES,
    ModelSpec,
    OPTIMIZERS,
    SpecError,
    Budget,
)
from .improver.meta_env import (
    AutoRefineEnv,
    KNOBS,
    candidate_screening,
    search_quality_v04,
)
from .improver.policy import SearchPolicy
from .improver.bandit import BanditPolicy
from .improver.rl_policy import MetaRLPolicy, train_policy, train_multi_policy
from .improver.curriculum import (  # 46.2 (v0.32): the three ladders
    CartPoleCurriculum,
    ParityCurriculum,
    SineCurriculum,
)
from .models.trees import TreeEnsemble, BoostingEnsemble
from .models.knn import KNN
from .models.convnet import ConvNet
from .pareto import ParetoFrontier
from .plotting import (
    ascii_pareto,
    ascii_score_curve,
    html_report,
    svg_pareto,
    svg_score_curve,
)
from .plugins import PluginError, discover, register_policies, register_tasks
from .tasks import (
    TASKS,
    AudioTask,
    CartPoleV1,
    CsvTask,
    GridNavV1,
    ImageTask,
    Parity4V1,
    ParityTask,
    SineRegressionV1,
    TextTask,
)

# SPEC.md 33.1 (C1): single version source — must equal pyproject.toml's
# [project].version (enforced by the A23 test); one step per feature round
# (v0.33 ⇒ 0.33.0, M36, SPEC.md 46)
__version__ = "0.33.0"

__all__ = [
    "AutoRefineEnv",
    "DashboardRunner",
    "SearchPolicy",
    "BanditPolicy",
    "search_quality_v04",
    "candidate_screening",
    "KNOBS",
    "MetaRLPolicy",
    "train_policy",
    "train_multi_policy",
    "ParityCurriculum",
    "SineCurriculum",
    "CartPoleCurriculum",
    "ascii_score_curve",
    "svg_score_curve",
    "ascii_pareto",
    "svg_pareto",
    "html_report",
    "PluginError",
    "discover",
    "register_policies",
    "register_tasks",
    "ModelSpec",
    "Budget",
    "DEFAULT_SPEC",
    "SpecError",
    "ACTIVATIONS",
    "BATCH_SIZES",
    "HIDDEN_LAYER_SIZES",
    "LR_SCHEDULES",
    "MODEL_FAMILIES",
    "OPTIMIZERS",
    "TreeEnsemble",
    "BoostingEnsemble",
    "KNN",
    "ConvNet",
    "ParetoFrontier",
    "TASKS",
    "CartPoleV1",
    "CsvTask",
    "GridNavV1",
    "Parity4V1",
    "ParityTask",
    "SineRegressionV1",
    "TextTask",
    "__version__",
]
