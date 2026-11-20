"""Qwen3.5-architecture support (``model_type: qwen3_5``; Qwen3.5 and Qwen3.6) for ``server.py``.

A model-support module provides ``MODEL_TYPE``, ``load_merged`` (API server process, CPU) and
``neuron_model_class`` (worker processes). It must not import vLLM or torch at module scope.
"""

MODEL_TYPE = "qwen3_5"


def load_merged(base_path, adapter_path):
    """Return a CPU Hugging Face model with the LoRA adapter merged, in the base's architecture."""
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(base_path)
    is_vlm = getattr(config, "vision_config", None) is not None
    if is_vlm:
        # vLLM replaces AutoConfig's Qwen3.5 registration with its serving
        # config. CPU HF models need their native config, without changing
        # that process-global registration used by the live server.
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

        config = Qwen3_5Config.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_path, config=config.text_config if is_vlm else config, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).merge_and_unload(safe_merge=True)
    if not is_vlm:
        return merged
    # SkyRL trains the text backbone. Preserve the original VLM checkpoint
    # architecture and vision weights expected by vLLM-Neuron's loader.
    full = AutoModelForImageTextToText.from_pretrained(base_path, config=config, dtype=torch.bfloat16, device_map="cpu")
    full.model.language_model.load_state_dict(merged.model.state_dict(), strict=True)
    full.lm_head.load_state_dict(merged.lm_head.state_dict(), strict=True)
    return full


def neuron_model_class():
    """vLLM-Neuron's text model class; older builds install it as a top-level ``qwen3_5`` package."""
    try:
        from qwen3_5.model import Qwen3_5ForCausalLM
    except ModuleNotFoundError as error:
        if error.name != "qwen3_5":
            raise
        from vllm_neuron.model.qwen3_5.model import Qwen3_5ForCausalLM
    return Qwen3_5ForCausalLM
