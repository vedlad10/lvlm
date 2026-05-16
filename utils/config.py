"""Configuration utilities for LVLM."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """Dictionary subclass that allows attribute access to values."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        
    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        super().__setitem__(key, value)
        self.__dict__[key] = value


def load_config(config_path: str) -> ConfigDict:
    """Load configuration from YAML file with variable interpolation.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ConfigDict with configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Interpolate variables
    config_dict = _interpolate_variables(config_dict)
    
    return ConfigDict(config_dict)


def save_config(config: ConfigDict, output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert ConfigDict to regular dict
    config_dict = _convert_to_dict(config)
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to: {output_path}")


def _interpolate_variables(obj: Any, root: Optional[Dict] = None) -> Any:
    """Recursively interpolate variables in configuration.
    
    Supports:
    - ${key}: Direct variable reference
    - ${key:format}: Formatted variable (e.g., ${now:%Y%m%d_%H%M%S})
    """
    if root is None:
        root = obj if isinstance(obj, dict) else {}
    
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = _interpolate_variables(value, root)
        return result
    
    elif isinstance(obj, list):
        return [_interpolate_variables(item, root) for item in obj]
    
    elif isinstance(obj, str):
        # Handle ${key} and ${key:format}
        import re
        pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            var_format = match.group(2)
            
            # Special handling for 'now'
            if var_name == 'now':
                now = datetime.now()
                return now.strftime(var_format) if var_format else str(now)
            
            # Look up variable in root config
            if var_name in root:
                value = root[var_name]
                return str(value)
            
            # Return unchanged if not found
            return match.group(0)
        
        return re.sub(pattern, replace_var, obj)
    
    return obj


def _convert_to_dict(obj: Any) -> Any:
    """Recursively convert ConfigDict and other objects to regular dicts."""
    if isinstance(obj, ConfigDict):
        return {k: _convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: _convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_dict(item) for item in obj]
    else:
        return obj


def merge_configs(base_config: ConfigDict, override_config: Dict) -> ConfigDict:
    """Merge override configuration into base configuration.
    
    Args:
        base_config: Base configuration
        override_config: Configuration overrides (supports dot notation like 'model.temporal_binding.enabled')
        
    Returns:
        Merged configuration
    """
    merged = ConfigDict(_convert_to_dict(base_config))
    
    for key_path, value in override_config.items():
        keys = key_path.split('.')
        current = merged
        
        # Navigate to the correct nested level
        for key in keys[:-1]:
            if key not in current:
                current[key] = ConfigDict()
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    return merged


def print_config(config: ConfigDict, indent: int = 0) -> None:
    """Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, (dict, ConfigDict)):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def get_config_path(config_name: str, config_dir: str = "configs") -> Path:
    """Get path to configuration file.
    
    Args:
        config_name: Name of config file (with or without .yaml extension)
        config_dir: Directory containing configs
        
    Returns:
        Path to configuration file
    """
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'
    
    config_path = Path(config_dir) / config_name
    return config_path
