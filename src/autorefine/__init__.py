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
    spec_n_params,  # 58.2 (v0.44): the spec's parameter count (size axis)
)
from .diagnostics import holdout_difficulty  # 58.1 (v0.44): the difficulty ranking
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
    TOKENS,
    ascii_pareto,
    ascii_score_curve,
    html_cover,
    html_report,
    resolve_tokens,
    svg_pareto,
    svg_score_curve,
    svg_bandit_beliefs,  # 57.4 (v0.43) the bandit belief bars
    svg_field_response,  # 57.2 (v0.43) the per-field response surfaces
    svg_gate_region,  # 57.3 (v0.43) the gate-decision region plot
    svg_spec_lineage,  # 57.1 (v0.43) the spec-lineage DAG
    svg_difficulty_ranking,  # 58.1 (v0.44) the difficulty-ranking lollipops
    svg_frontier3,  # 58.2 (v0.44) the 3-objective frontier (two panels)
    svg_whatif_effect,  # 60.1 (v0.46) the what-if estimated effect
    svg_spec_fingerprint,  # 60.2 (v0.46) the spec fingerprint ("DNA")
    svg_interaction_heatmap,  # 60.3 (v0.46) the interaction heatmap
    svg_weighted_reslice,  # 60.4 (v0.46) the objective-weight reslice
)
from .dossier import build_dossier  # 58.3 (v0.44) the run dossier
from .provenance import (
    canonical_hash,
    env_provenance,
    provenance_card,
    provenance_payload,
    svg_provenance,
)
from .research import (  # 57 (v0.43): the research decision surfaces (pure)
    bandit_beliefs,
    field_response_stats,
    gate_region_candidates,
    spec_lineage,
    frontier3,  # 58.2 (v0.44) the 3-objective frontier
)
from .steering import (  # 59 (v0.45): parameters interpreted + modified
    SteeringState,
    apply_steering,
    manual_spec,
    parameter_inspection,
    train_manual,
)
from .whatif import (  # 60 (v0.46): parameters what-iffed + compared
    fingerprint_diff,
    interaction_matrix,
    spec_fingerprint,
    whatif_preview,
    weighted_reslice,
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
# (v0.46 ⇒ 0.46.0, M49, SPEC.md 60)
__version__ = "0.46.0"

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
    "TOKENS",
    "html_cover",
    "resolve_tokens",
    "svg_bandit_beliefs",
    "svg_field_response",
    "svg_gate_region",
    "svg_spec_lineage",
    "svg_difficulty_ranking",
    "svg_frontier3",
    "build_dossier",
    "holdout_difficulty",
    "frontier3",
    "spec_n_params",
    "SteeringState",
    "apply_steering",
    "manual_spec",
    "parameter_inspection",
    "train_manual",
    "whatif_preview",
    "spec_fingerprint",
    "fingerprint_diff",
    "interaction_matrix",
    "weighted_reslice",
    "svg_whatif_effect",
    "svg_spec_fingerprint",
    "svg_interaction_heatmap",
    "svg_weighted_reslice",
    "canonical_hash",
    "env_provenance",
    "provenance_card",
    "provenance_payload",
    "svg_provenance",
    "bandit_beliefs",
    "field_response_stats",
    "gate_region_candidates",
    "spec_lineage",
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
