from __future__ import annotations

from pathlib import Path
from path_tool import get_abs_path
from config_handler import prompt_config, skill_config


def load_system_prompts() -> dict[str, str]:
    """
    加载系统提示词
    """
    try:
        system_prompt_path = get_abs_path(prompt_config["system_prompt_path"])
    except KeyError as e:
        raise e

    try:
        return Path(system_prompt_path).read_text(encoding="utf-8")
    except Exception as e:
        raise e

def load_verify_prompts() -> dict[str, str]:
    """
    加载验证提示词
    """
    try:
        system_prompt_path = get_abs_path(prompt_config["system_prompt_path"])
    except KeyError as e:
        raise e

    try:
        return Path(system_prompt_path).read_text(encoding="utf-8")
    except Exception as e:
        raise e
    
def load_memory_proposal_skill():
    """
    加载记忆提案的skill
    """
    try:
        memory_proposal_path = get_abs_path(skill_config["memory_proposal_workflow"])
    except KeyError as e:
        raise e

    try:
        return Path(memory_proposal_path).read_text(encoding="utf-8")
    except Exception as e:
        raise e

if __name__ == '__main__':
    print(load_memory_proposal_skill())
