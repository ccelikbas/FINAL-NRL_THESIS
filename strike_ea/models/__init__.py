"""Model architectures for Strike-EA."""

from .actor import make_actor
from .critic import make_critic

__all__ = ["make_actor", "make_critic"]
