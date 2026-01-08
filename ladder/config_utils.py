"""
Utility module for loading gin configs and dynamic imports.
"""
import os
import sys
import importlib.util


def load_gin_config(config_path=None):
    """
    Load gin configuration file and return parsed parameters.
    
    Args:
        config_path: Path to gin config file. If None, uses gconfigs/base.gin
    
    Returns:
        dict: Parsed configuration parameters
    """
    if config_path is None:
        # Use base config
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'gconfigs', 'base.gin')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse key = value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                config[key] = value
    
    return config


def setup_pi3_path(config_path=None):
    """
    Load gin config and add pi3 path to sys.path for dynamic imports.
    
    Args:
        config_path: Path to gin config file. If None, uses gconfigs/base.gin
    
    Returns:
        dict: The full config dictionary
    """
    config = load_gin_config(config_path)
    
    pi3_path = config.get('pi3_path', '.')
    
    # Convert relative path to absolute
    if not os.path.isabs(pi3_path):
        # Resolve relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pi3_path = os.path.join(project_root, pi3_path)
    
    pi3_path = os.path.abspath(pi3_path)
    
    # Add to sys.path if not already there
    if pi3_path not in sys.path:
        sys.path.insert(0, pi3_path)
    
    return config


def dynamic_import(module_name, package_path=None):
    """
    Dynamically import a module from a specific path.
    
    Args:
        module_name: Full module name (e.g., 'pi3.utils.basic')
        package_path: Path to the package root. If None, uses configured pi3_path
    
    Returns:
        module: The imported module
    """
    if package_path is None:
        # Use the pi3_path from config that was already set up
        parts = module_name.split('.')
        if parts[0] == 'pi3':
            # Module is already in sys.path, just import normally
            return __import__(module_name, fromlist=[parts[-1]])
    else:
        # Custom path provided
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
        parts = module_name.split('.')
        return __import__(module_name, fromlist=[parts[-1]])
