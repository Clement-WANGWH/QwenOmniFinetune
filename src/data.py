import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict
import copy
from qwen_vl_utils import process_vision_info
import torchaudio
import numpy as np

AUDIO_TOKEN = "<|audio_start|><|audio_end|>"
IGNORE_INDEX = -100


class OmniDataset(Dataset):

    def __init__(self, data_path, processor, data_args):
        super().__init__()
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.processor = processor
        self.data_args = data_args
        self.media_folder = data_args.media_folder

    def __len__(self):
        return len(self.data)

    def process_audio(self, audio_path):
        full_path = os.path.join(self.media_folder, audio_path)

        waveform, sample_rate = torchaudio.load(full_path)

        if sample_rate != self.data_args.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.data_args.audio_sample_rate
            )
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        max_samples = int(self.data_args.max_audio_length * self.data_args.audio_sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[1]))

        return waveform

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        images = []
        videos = []
        audio_features = None

        if "image" in item:
            image_files = item["image"] if isinstance(item["image"], list) else [item["image"]]
            for img_file in image_files:
                img_path = os.path.join(self.media_folder, img_file)
                images.append(self.processor.image_processor(img_path))

        if "video" in item:
            video_file = item["video"]
            video_path = os.path.join(self.media_folder, video_file)
            videos.append(self.processor.video_processor(video_path, fps=self.data_args.fps))

            if item.get("has_audio", False):
                audio_features = self.process_audio(video_file)

        if "audio" in item:
            audio_features = self.process_audio(item["audio"])

        conversation = []
        for conv in item["conversations"]:
            content = []

            if conv["from"] == "human":
                if images:
                    for _ in images:
                        content.append({"type": "image", "image": "<image>"})
                if videos:
                    content.append({"type": "video", "video": "<video>"})
                if audio_features is not None:
                    content.append({"type": "text", "text": AUDIO_TOKEN})

            content.append({"type": "text", "text": conv["value"]})
            conversation.append({"role": conv["from"], "content": content})

        text = self.processor.apply_chat_template(conversation, tokenize=False)
        inputs = self.processor(
            text=text,
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors="pt",
            padding=False
        )

        if audio_features is not None:
            inputs["audio_features"] = audio_features

        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            **{k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
               for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}
        }


class DataCollator:
    """Data collator for multimodal inputs"""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate different types of inputs
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad sequences
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for ids, labs in zip(input_ids, labels):
            padding_length = max_length - len(ids)
            padded_input_ids.append(
                torch.cat([ids, torch.full((padding_length,), self.pad_token_id)])
            )
            padded_labels.append(
                torch.cat([labs, torch.full((padding_length,), IGNORE_INDEX)])
            )
            attention_masks.append(
                torch.cat([torch.ones_like(ids), torch.zeros(padding_length)])
            )

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
        }

        # Handle other features (images, videos, audio)
        for key in features[0].keys():
            if key not in ["input_ids", "labels", "attention_mask"]:
                if all(key in f for f in features):
                    values = [f[key] for f in features]
                    if isinstance(values[0], torch.Tensor):
                        batch[key] = torch.stack(values)
                    else:
                        batch[key] = values

        return batch


def make_data_module(model_id, processor, data_args):
    """Create dataset and data collator"""
    dataset = OmniDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args
    )

    data_collator = DataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator
    )