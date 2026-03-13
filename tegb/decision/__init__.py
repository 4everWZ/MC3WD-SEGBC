from tegb.config.schema import DecisionSection

from .adaptive_three_way import AdaptiveThreeWayDecider
from .spherical_collision_three_way import SphericalCollisionThreeWayDecider
from .three_way import ThreeWayDecider


def build_decider(cfg: DecisionSection):
    if cfg.mode == "legacy_threshold":
        return ThreeWayDecider(cfg)
    if cfg.mode == "v3_spherical_collision":
        return SphericalCollisionThreeWayDecider(cfg)
    return AdaptiveThreeWayDecider(cfg)


__all__ = [
    "ThreeWayDecider",
    "AdaptiveThreeWayDecider",
    "SphericalCollisionThreeWayDecider",
    "build_decider",
]
