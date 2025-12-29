# config/__init__.py
"""
Dheera Configuration Module

Configuration files:
- dheera_config.yaml - Main configuration
- identity.yaml - Agent identity and personality
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not os.path.exists(path):
        return {}
    
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "version": "0.3.0",
        "slm": {
            "provider": "ollama",
            "model": "phi3:mini",
            "temperature": 0.7,
            "max_tokens": 512,
        },
        "dqn": {
            "state_dim": 64,
            "action_dim": 8,
            "hidden_dim": 128,
            "gamma": 0.99,
            "lr": 1e-4,
        },
        "database": {
            "path": "dheera.db",
        },
    }


__all__ = [
    "load_config",
    "save_config",
    "get_default_config",
]
