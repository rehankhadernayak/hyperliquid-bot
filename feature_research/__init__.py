"""Feature research package — offline predictive feature analysis."""

from feature_research.feature_config import FeatureResearchConfig
from feature_research.feature_engine import FeatureResearchEngine, FeatureResearchError
from feature_research.feature_safety import PAPER_ONLY, SAFETY_STATEMENT

__all__ = [
    "FeatureResearchConfig",
    "FeatureResearchEngine",
    "FeatureResearchError",
    "PAPER_ONLY",
    "SAFETY_STATEMENT",
]
