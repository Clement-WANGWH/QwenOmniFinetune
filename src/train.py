import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from .data import make_data_module
from .trainer import OmniTrainer
from .utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3


@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen/Qwen2.5-Omni-7B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the training data JSON file"})
    media_folder: str = field(metadata={"help": "Path to media files (images/videos/audio)"})
    image_min_pixels: int = field(default=3136)
    image_max_pixels: int = field(default=12845056)
    video_min_pixels: int = field(default=100352)
    video_max_pixels: int = field(default=602112)
    fps: float = field(default=1.0)
    audio_sample_rate: int = field(default=16000)
    max_audio_length: float = field(default=30.0)  # seconds


@dataclass
class TrainingArguments:
    output_dir: str = field(metadata={"help": "Output directory"})
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=1e-4)
    vision_lr: float = field(default=2e-6)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    weight_decay: float = field(default=0.1)

    # LoRA specific
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[str] = field(default=None)

    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    deepspeed: str = field(default="scripts/deepspeed_configs/zero3.json")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=10)
    logging_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="tensorboard")
    use_liger: bool = field(default=True)
    local_rank: int = field(default=-1)


def find_target_modules(model, exclude_patterns=["visual", "audio_encoder"]):
    target_modules = []
    for name, module in model.named_modules():
        if any(pattern in name for pattern in exclude_patterns):
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            target_modules.append(name)
    return list(set(target_modules))


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.use_liger:
        apply_liger_kernel_to_qwen2_5_vl()

    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2"
    )

    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Configure LoRA
    target_modules = find_target_modules(model)
    if training_args.lora_target_modules:
        target_modules = training_args.lora_target_modules.split(",")

    peft_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_args.lora_dropout,
        bias="none"
    )

    model = get_peft_model(model, peft_config)

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    data_module = make_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args
    )

    trainer = OmniTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )

    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    trainer.save_state()

    state_dict = get_peft_state_maybe_zero_3(model.named_parameters())
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters(), require_grad_only=True
    )

    if training_args.local_rank in [0, -1]:
        model.config.save_pretrained(training_args.output_dir)
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        torch.save(non_lora_state_dict,
                   os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))


if __name__ == "__main__":
    main()