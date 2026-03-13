from tegb.config.schema import GranularSection

from .probabilistic_builder import ProbabilisticGranularBuilder
from .spherical_entropy_builder import SphericalEntropyBuilder
from .topological_builder import TopologicalGranularBuilder


def build_granular_builder(cfg: GranularSection, random_state: int = 42):
    if cfg.mode == "legacy_sphere":
        return TopologicalGranularBuilder(cfg, random_state=random_state)
    if cfg.mode == "v3_spherical_entropy":
        return SphericalEntropyBuilder(cfg, random_state=random_state)
    return ProbabilisticGranularBuilder(cfg, random_state=random_state)


__all__ = [
    "TopologicalGranularBuilder",
    "ProbabilisticGranularBuilder",
    "SphericalEntropyBuilder",
    "build_granular_builder",
]
