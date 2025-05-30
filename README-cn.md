[🇺🇸 English Guide](README.md)

# MiMo-VL-7B-RL 快速API调用脚本说明

本项目中的 `MiMoVLM-api_server.py` 脚本用于快速部署和调用小米开源的多模态视觉语言模型（VLM）—— **MiMo-VL-7B-RL**，支持图片描述等多模态推理任务。

## 模型简介

MiMo-VL-7B-RL 是由小米大模型团队发布的高性能视觉语言模型，具备强大的图像理解、推理和多模态对话能力。该模型采用原生分辨率ViT编码器、MLP投影器和MiMo-7B语言模型，经过多阶段预训练和混合强化学习优化，在多项公开基准上取得领先表现。

- **模型主页与技术报告**：[小米MiMo-VL GitHub](https://github.com/XiaomiMiMo/MiMo-VL/tree/main)
- **模型权重下载**：[ModelScope下载地址](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-VL-7B-RL/files)

## 主要功能

- 支持通过图片URL或本地上传图片进行视觉描述生成
- 支持自定义提示词（prompt）
- 提供标准RESTful API接口，便于集成到各类应用
- 自动管理临时文件，支持高并发调用

## 快速开始

1. **准备模型权重**
   - 请从 [ModelScope下载页面](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-VL-7B-RL/files) 下载模型权重，并解压到 `MiMoVLM-api_server.py` 中 `MODEL_PATH` 指定的目录（默认 `/hy-tmp/data/MiMo-VL-7B-RL`）。

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **启动API服务**
   ```bash
   python MiMoVLM-api_server.py
   ```
   启动后，服务监听在 `http://0.0.0.0:8000`。

## API接口说明

### 1. 图片URL描述
- **接口地址**：`POST /describe_url/`
- **请求体**：
  ```json
  {
    "image_url": "图片的URL地址",
    "prompt_text": "（可选）自定义提示词"
  }
  ```
- **返回示例**：
  ```json
  {
    "description": "图片内容描述",
    "prompt_used": "实际使用的提示词",
    "error": null
  }
  ```

### 2. 图片文件上传描述
- **接口地址**：`POST /describe_upload/`
- **请求体**：
  - `image`：上传的图片文件（form-data）
  - `prompt_text`：可选，自定义提示词
- **返回同上**

## 依赖环境
详见 `requirements.txt`。

## 致谢
本项目基于小米大模型团队开源的 [MiMo-VL](https://github.com/XiaomiMiMo/MiMo-VL/tree/main) 项目，特此致谢！

如需了解更多技术细节，请参考 [MiMo-VL 技术报告](https://github.com/XiaomiMiMo/MiMo-VL/blob/main/MiMo-VL-Technical-Report.pdf)。 