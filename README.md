[🇨🇳 中文说明](README-cn.md) | **🇺🇸 English**

# MiMo-VL-7B-RL Quick API Script Guide

The `MiMoVLM-api_server.py` script in this project is designed for rapid deployment and invocation of Xiaomi's open-source multi-modal Vision-Language Model (VLM) — **MiMo-VL-7B-RL**. It supports image captioning and other multi-modal reasoning tasks.

## Model Overview

MiMo-VL-7B-RL is a high-performance vision-language model released by Xiaomi's large model team. It features powerful image understanding, reasoning, and multi-modal dialogue capabilities. The model utilizes a native-resolution ViT encoder, MLP projector, and MiMo-7B language model, and is optimized through multi-stage pre-training and mixed reinforcement learning, achieving state-of-the-art results on several public benchmarks.

- **Model Homepage & Technical Report**: [Xiaomi MiMo-VL GitHub](https://github.com/XiaomiMiMo/MiMo-VL/tree/main)
- **Model Weights Download**: [ModelScope Download Link](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-VL-7B-RL/files)

## Main Features

- Supports image captioning via image URL or local file upload
- Customizable prompt (instruction) support
- Standard RESTful API interface for easy integration
- Automatic management of temporary files, suitable for high concurrency

## Quick Start

1. **Prepare Model Weights**
   - Download the model weights from [ModelScope Download Page](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-VL-7B-RL/files) and extract them to the directory specified by `MODEL_PATH` in `MiMoVLM-api_server.py` (default: `/hy-tmp/data/MiMo-VL-7B-RL`).

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API Service**
   ```bash
   python MiMoVLM-api_server.py
   ```
   The service will listen on `http://0.0.0.0:8000` after startup.

## API Endpoints

### 1. Image URL Captioning
- **Endpoint**: `POST /describe_url/`
- **Request Body**:
  ```json
  {
    "image_url": "URL of the image",
    "prompt_text": "(Optional) Custom prompt"
  }
  ```
- **Response Example**:
  ```json
  {
    "description": "Image caption",
    "prompt_used": "Prompt actually used",
    "error": null
  }
  ```

### 2. Image File Upload Captioning
- **Endpoint**: `POST /describe_upload/`
- **Request Body**:
  - `image`: The uploaded image file (form-data)
  - `prompt_text`: (Optional) Custom prompt
- **Response**: Same as above

## Dependencies
See `requirements.txt` for details.

## Acknowledgement
This project is based on the open-source [MiMo-VL](https://github.com/XiaomiMiMo/MiMo-VL/tree/main) project by Xiaomi Large Model Team. Special thanks!

For more technical details, please refer to the [MiMo-VL Technical Report](https://github.com/XiaomiMiMo/MiMo-VL/blob/main/MiMo-VL-Technical-Report.pdf). 