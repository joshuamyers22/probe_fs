"""Public API for probe-gated feature selection."""

from .config import MethodSpec
from .experiments import StudyConstraints, run_spec
from .importance import ImportanceResult, estimate_importance
from .metrics import SignalClasses
from .nested import NestedResult, finalize_for_deployment, nested_evaluate
from .players import Players
from .simulate import SimData, make_primary, make_secondary

__all__ = [
    "ImportanceResult",
    "MethodSpec",
    "NestedResult",
    "Players",
    "SignalClasses",
    "SimData",
    "StudyConstraints",
    "estimate_importance",
    "finalize_for_deployment",
    "make_primary",
    "make_secondary",
    "nested_evaluate",
    "run_spec",
]
