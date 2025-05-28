import os
import torch
from transformers import Trainer
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
from .utils import get_peft_state_non_lora_maybe_zero_3


class OmniTrainer(Trainer):
    """Custom trainer for Qwen2.5-Omni with multi-learning rate support"""

    def create_optimizer(self):
        """Setup optimizer with different learning rates for vision and language components"""
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # Separate vision and language parameters
            vision_params = []
            language_params = []

            for name, param in opt_model.named_parameters():
                if param.requires_grad:
                    if "visual" in name or "audio_encoder" in name:
                        vision_params.append(name)
                    else:
                        language_params.append(name)

            # Create parameter groups
            optimizer_grouped_parameters = []

            # Language model parameters
            optimizer_grouped_parameters.extend([
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in language_params and n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if n in language_params and n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                }
            ])

            # Vision/audio parameters (if any are trainable)
            if vision_params:
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in vision_params and n in decay_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in vision_params and n not in decay_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_lr,
                    }
                ])

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        """Save LoRA checkpoint with non-LoRA weights"""
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        self.save_model(output_dir, _internal_call=True)

        # Save non-LoRA weights
        non_lora_weights = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters(), require_grad_only=False
        )
        torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

        # Save optimizer and scheduler
        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir)
            self._save_rng_state(output_dir)

        # Save trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))