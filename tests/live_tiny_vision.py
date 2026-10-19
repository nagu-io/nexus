"""A tiny, randomly initialised Qwen3-VL for testing NEXUS Vision's training and loading code.

It has the real model's architecture and token layout (the same chat template,
special tokens, patch size and merge size) at a size a CPU trains in seconds, so
the fine-tuning, model card, adapter loading and evaluation code all run for
real without a download or a GPU. Its answers are noise; the tests check the
plumbing, not what it learns.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]


def vision_stack_available() -> bool:
    try:
        import peft  # noqa: F401
        import PIL  # noqa: F401
        import tokenizers  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return hasattr(transformers, "Qwen3VLForConditionalGeneration")


def build_tiny_qwen3_vl(folder: Path) -> Path:
    """Write a tiny Qwen3-VL model and its processor into ``folder``; returns the folder."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        Qwen2Tokenizer,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )

    from nexus.live.model import OBSERVE_PROMPT, request_text, system_prompt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # A byte-level BPE tokenizer like Qwen's, trained on the text NEXUS Live sends and receives.
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    corpus = [
        system_prompt("normalized"),
        OBSERVE_PROMPT,
        request_text('Click "Save"', ["click at (500, 250)"], [("user", "hello"), ("assistant", "hi")]),
        '{"thought": "", "say": "Done.", "action": "done"}',
        '{"thought": "", "say": "", "action": "click", "x": 512, "y": 384}',
        '{"thought": "", "say": "", "action": "type", "text": "hello world"}',
        '{"thought": "", "say": "", "action": "hotkey", "keys": ["ctrl", "s"]}',
        "assistant user system 0123456789",
    ] * 4
    trainer = trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    bpe.train_from_iterator(corpus, trainer)
    vocab = bpe.get_vocab()
    merges = _merges(bpe)
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=merges, eos_token="<|im_end|>", pad_token="<|endoftext|>", unk_token=None)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS}

    from transformers import Qwen2VLImageProcessor, Qwen3VLVideoProcessor

    image_processor = Qwen2VLImageProcessor(
        patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3,
        size={"shortest_edge": 32 * 32, "longest_edge": 16777216},
    )
    video_processor = Qwen3VLVideoProcessor(
        patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3,
    )
    template = (FIXTURES / "qwen3_vl_chat_template.jinja").read_text(encoding="utf-8")
    processor = Qwen3VLProcessor(
        image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor, chat_template=template
    )

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 4096,
            "rope_theta": 5000000,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
            "tie_word_embeddings": True,
            "bos_token_id": ids["<|endoftext|>"],
            "eos_token_id": ids["<|im_end|>"],
            "pad_token_id": ids["<|endoftext|>"],
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=ids["<|image_pad|>"],
        video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"],
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(config)
    model.generation_config.eos_token_id = [ids["<|im_end|>"], ids["<|endoftext|>"]]
    model.generation_config.pad_token_id = ids["<|endoftext|>"]
    model.save_pretrained(folder)
    processor.save_pretrained(folder)
    return folder


def _merges(bpe) -> list[tuple[str, str]]:
    """The trained merges, as (left, right) pairs: read back from the tokenizer's saved form."""
    import json

    merges = json.loads(bpe.to_str())["model"]["merges"]
    return [tuple(merge.split(" ", 1)) if isinstance(merge, str) else tuple(merge) for merge in merges]
