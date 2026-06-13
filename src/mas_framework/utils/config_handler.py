import yaml
from mas_framework.utils.path_tool import get_abs_path

_DEFAULT_PROMPT_PATH = get_abs_path("configs/prompts.yml")
_DEFAULT_SKILL_PATH = get_abs_path("configs/skills.yml")

def load_prompts_config(config_path: str=_DEFAULT_PROMPT_PATH, encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_skills_config(config_path: str=_DEFAULT_SKILL_PATH, encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

skill_config = load_skills_config()
prompt_config = load_prompts_config()
