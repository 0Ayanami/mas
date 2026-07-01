from __future__ import annotations

from pathlib import Path
from mas_framework.utils.path_tool import get_abs_path
from mas_framework.utils.config_handler import prompt_config, skill_config


def load_system_prompts() -> str:
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

def load_verify_prompts() -> str:
    """
    加载验证提示词
    """
    try:
        verify_prompt_path = get_abs_path(prompt_config["verify_prompt_path"])
    except KeyError as e:
        raise e

    try:
        return Path(verify_prompt_path).read_text(encoding="utf-8")
    except Exception as e:
        raise e
    
def load_propose_prompts() -> str:
    """
    加载提案提示词
    """
    try:
        propose_prompt_path = get_abs_path(prompt_config["propose_prompt_path"])
    except KeyError as e:
        raise e

    try:
        return Path(propose_prompt_path).read_text(encoding="utf-8")
    except Exception as e:
        raise e

def load_create_proposal_prompts() -> str:
    """
    加载创建提案提示词
    """
    try:
        create_proposal_prompt_path = get_abs_path(prompt_config["create_proposal_prompt_path"])
    except KeyError as e:
        raise e

    try:
        return Path(create_proposal_prompt_path).read_text(encoding="utf-8")
    except Exception as e:
        raise e
    
def load_memory_proposal_skill() -> str:
    """
    加载记忆提案的skill
    """
    try:
        memory_proposal_path = get_abs_path(
            skill_config.get("Memory_Proposal_Skill")
            or skill_config["memory_proposal_skill"]
        )
    except KeyError as e:
        raise e

    try:
        return Path(memory_proposal_path).read_text(encoding="utf-8")
    except Exception as e:
        raise e

if __name__ == '__main__':
    print(load_memory_proposal_skill())
