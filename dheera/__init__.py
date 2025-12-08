# dheera/__init__.py

from .core import Dheera

def create_dheera(
    config_path: str | None = None,
    debug: bool = False,
    fresh: bool = False,  # <--- added
) -> Dheera:
    """
    Factory to create a Dheera instance.

    Args:
        config_path: Optional path to config YAML.
        debug: Enable debug mode.
        fresh: If True, start with a fresh DQN / no prior checkpoint.
               Currently optional – wiring is inside this function.
    """
    # your existing config loading, etc.
    # e.g.
    # config = load_config(config_path)
    # dheera = Dheera(config=config, debug=debug)

    dheera = Dheera(config_path=config_path, debug=debug)  # or whatever you already have

    # Optional: handle 'fresh' semantics here
    if fresh:
        # Don’t auto-load previous checkpoint.
        # If your Dheera class auto-loads in __init__, you might want to clear or skip it.
        try:
            # Example patterns – pick what exists in your class:
            # dheera.clear_memory()
            # dheera.dqn.reset()
            # or just ensure you don’t call load_checkpoint() when fresh=True
            pass
        except Exception:
            # Ignore if not implemented yet
            pass
    else:
        # If you previously loaded checkpoints here, keep that behavior:
        try:
            dheera.load_checkpoint()
        except Exception:
            # First run / no checkpoint yet
            pass

    return dheera

