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
from .improver.rl_policy import (
    MetaRLPolicy,
    train_policy,
    train_multi_policy,
    save_policy,  # 67.1 (v0.53): policy persistence
    load_policy,
    policy_to_bytes,   # 68.2.1 (v0.54): the in-memory policy bytes
    policy_from_bytes,  # 68.2.1 (v0.54): reconstruct a policy from bytes
)
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
    svg_run_verdict,  # the run verdict card (target vs final best)
    svg_knob_signal,  # the decisive-knob ranking (signal vs noise)
    svg_efficiency_knee,  # the efficiency knee (bang for buck)
    svg_rejection_anatomy,  # the rejection mix + stall story
    svg_is_well_formed,  # 69.3 (v0.55) the SVG render guard
    visual_card,  # 71.1.2 (v0.57, A61) the visual-card wrapper
    VISUAL_CSS,  # 71.1 (v0.57, A61) the shared visual-card stylesheet
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
    run_verdict,  # the run verdict card's data
    knob_signal,  # the decisive-knob ranking's data
    frontier_knee,  # the efficiency knee's data
    rejection_anatomy,  # the rejection mix + stall story's data
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
from .calibration import (  # 61.1.3 / 61.4 (v0.47): constraint / calibration metrics
    coverage_width,
    ece,
    ensemble_ece,
    monotonicity_slope,
    per_class_f1,
)
from .gate import compute_actuals  # 61.1.2 (v0.47): the richer actuals producer
from .scenarios import (  # 61.3 (v0.47): regime / stress scenarios (pure config)
    Scenario,
    drifting,
    fewshot,
    trap,
)
from .audience import (  # 62 (v0.48, B1): the audience axis + verify
    AUDIENCES,
    build_view,
    data_fingerprint,
    domain_view,
    exec_view,
    html_view,
    regulator_view,
    render_verify,
    render_view,
    technical_view,
    verify_run,
)
from .reporting import (  # 63 (v0.49, B2/B3/B4): formats + user guide + decision
    REPORT_FORMATS,
    benchmark_view,      # 64.1 (v0.50, B5): the longitudinal report
    build_report_doc,
    decision_view,
    render_benchmark,    # 64.1.4 (v0.50, B5)
    render_benchmark_md,  # 64.1.4 (v0.50, B5)
    render_decision,
    render_report,
    render_report_md,
    render_report_pdf,
    render_report_txt,
    render_user_guide,
    user_guide_view,
)
from .uncertainty import (  # 64.2 (v0.50, B6) + 66 (v0.52, B7): the reads
    format_pm,
    headline_uncertainty,
    seed_spread_stats,
    verdict_robustness,
)
from .quickstart import (  # 65 (v0.51): the Quickstart (one source, three surfaces)
    QUICKSTART_INTRO,
    QUICKSTART_STEPS,
    demo_recipe,
    quickstart_commands,
    quickstart_steps,
    render_quickstart,
    render_quickstart_md,
    run_demo,
)
from .workflow import (  # 69.4 (v0.55): the procedural workflow (state + strip + next steps)
    STEP_CURRENT,
    STEP_DONE,
    STEP_TODO,
    WORKFLOW_STEPS,
    next_steps,
    svg_workflow_strip,
    workflow_state,
)
from .plugins import PluginError, discover, register_policies, register_tasks
from .tasks import (
    TASKS,
    AudioTask,
    CartPoleV1,
    CsvTask,
    FinanceForecastTask,
    GridNavV1,
    ImageTask,
    MedicalTabularTask,
    Parity4V1,
    ParityTask,
    RobustTask,
    SineRegressionV1,
    TextTask,
)
from .rl_dashboard import (  # 68 (v0.54, A58): the RL loop in the dashboard
    RLMultiRunner,
    RLRunner,
    describe_policy,
)

# SPEC.md 33.1 (C1): single version source — must equal pyproject.toml's
# [project].version (enforced by the A23 test); one step per feature round
# (v0.61 ⇒ 0.61.0, M64, SPEC.md 75)
__version__ = "0.61.0"

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
    "save_policy",
    "load_policy",
    "policy_to_bytes",
    "policy_from_bytes",
    "RLRunner",
    "RLMultiRunner",
    "describe_policy",
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
    "svg_run_verdict",
    "svg_knob_signal",
    "svg_efficiency_knee",
    "svg_rejection_anatomy",
    "canonical_hash",
    "env_provenance",
    "provenance_card",
    "provenance_payload",
    "svg_provenance",
    "bandit_beliefs",
    "field_response_stats",
    "gate_region_candidates",
    "spec_lineage",
    "run_verdict",
    "knob_signal",
    "frontier_knee",
    "rejection_anatomy",
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
    # v0.47: more specialized scenarios (SPEC.md 61, A51)
    "per_class_f1", "ece", "coverage_width", "monotonicity_slope",
    "ensemble_ece", "compute_actuals",
    "MedicalTabularTask", "FinanceForecastTask", "RobustTask",
    "Scenario", "fewshot", "drifting", "trap",
    # v0.48: the audience axis + verify (SPEC.md 62, A52)
    "AUDIENCES", "exec_view", "domain_view", "technical_view", "regulator_view",
    "build_view", "render_view", "html_view", "data_fingerprint",
    "verify_run", "render_verify",
    # v0.49: format breadth + the end-user guide + the decision artifact
    # (SPEC.md 63, A53)
    "REPORT_FORMATS", "build_report_doc", "render_report_md", "render_report_txt",
    "render_report_pdf", "render_report", "user_guide_view",
    "render_user_guide", "decision_view", "render_decision",
    "verdict_robustness",  # 66.1 (v0.52, B7): verdict robustness (A56)
    # v0.50: the benchmark / longitudinal report + uncertainty on headlines
    # (SPEC.md 64, A54). Note: the top-level `spec_fingerprint` stays the
    # v0.46 whatif "DNA" bars (one name, one binding); the B5 12-hex spec
    # identity is `autorefine.reporting.spec_fingerprint` (64.1.2), used
    # by `benchmark_view` internally.
    "benchmark_view", "render_benchmark", "render_benchmark_md",
    "seed_spread_stats", "format_pm", "headline_uncertainty",
    # v0.51: the Quickstart — one source, three surfaces (SPEC.md 65, A55)
    "QUICKSTART_INTRO", "QUICKSTART_STEPS", "quickstart_steps",
    "quickstart_commands", "render_quickstart", "render_quickstart_md",
    "demo_recipe", "run_demo",
    # v0.55: the procedural workflow core (SPEC.md 69, A59)
    "WORKFLOW_STEPS", "STEP_DONE", "STEP_CURRENT", "STEP_TODO",
    "workflow_state", "svg_workflow_strip", "next_steps",
    "svg_is_well_formed",
    # v0.57: the visual-card interface (SPEC.md 71, A61)
    "visual_card", "VISUAL_CSS",
    "__version__",
]
