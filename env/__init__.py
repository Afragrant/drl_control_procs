"""RL environment package for the pantograph-catenary simulator."""

from xuance.environment import REGISTRY_ENV

from .procs_core import ProcsCore
from .procs_env import ProcsEnv, ProcsXuanceEnv

REGISTRY_ENV['PROCS'] = ProcsXuanceEnv

__all__ = ["ProcsCore", "ProcsEnv", "ProcsXuanceEnv"]
