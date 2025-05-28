import argparse
import os
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def merge_lora_weights(model_path, model_base, save_path):
    """Merge LoRA weights with base model"""

    print(f"Loading base model from {model_base}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Loading LoRA weights...")
    # Load non-LoRA trainable parameters
    non_lora_weights = torch.load(
        os.path.join(model_path, "non_lora_state_dict.bin"),
        map_location="cpu"
    )
    model.load_state_dict(non_lora_weights, strict=False)

    # Load LoRA model
    model = PeftModel.from_pretrained(model, model_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {save_path}...")
    model.save_pretrained(save_path)

    # Save processor
    processor = AutoProcessor.from_pretrained(model_base)
    processor.save_pretrained(save_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-Omni-7B-Instruct")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save merged model")

    args = parser.parse_args()
    merge_lora_weights(args.model_path, args.model_base, args.save_path)