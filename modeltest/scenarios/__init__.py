"""Built-in test scenarios shipped with modeltest."""

from modeltest.scenarios.data import DataInvariantTest, NoNullTest
from modeltest.scenarios.drift import DataDriftTest, KSTest
from modeltest.scenarios.explainability import FeatureDominanceTest, TopFeaturesTest
from modeltest.scenarios.fairness import EqualOpportunityTest, StatisticalParityTest
from modeltest.scenarios.performance import GroupPerformanceTest, MinimumAccuracyTest
from modeltest.scenarios.robustness import RobustnessTest

__all__ = [
    "MinimumAccuracyTest",
    "GroupPerformanceTest",
    "RobustnessTest",
    "DataInvariantTest",
    "NoNullTest",
    "DataDriftTest",
    "KSTest",
    "EqualOpportunityTest",
    "StatisticalParityTest",
    "FeatureDominanceTest",
    "TopFeaturesTest",
]
