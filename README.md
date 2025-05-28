# 训练数据格式与依赖安装

## 训练数据格式

训练数据应为 JSON 格式，每个样本包含以下字段：

```json
[
  {
    "id": "sample_1",
    "image": "image.jpg",  
    "video": "video.mp4",  
    "audio": "audio.wav",  
    "has_audio": true,
    "conversations": [
      {
        "from": "human",
        "value": "描述这个图片/视频/音频"
      },
      {
        "from": "assistant",
        "value": "这是一段..."
      }
    ]
  }
]
```
## 依赖安装
请使用以下命令安装所需的依赖：
```bash
pip install flash-attn --no-build-isolation
