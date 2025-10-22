import yaml

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
