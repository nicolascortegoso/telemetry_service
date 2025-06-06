import yaml
from typing import Any, Dict

def get_config() -> Dict[str, Any]:
    """
    Load configuration settings from the 'config.yaml' file.

    Returns:
        Dict[str, Any]: Parsed configuration data as a dictionary.
    """
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config