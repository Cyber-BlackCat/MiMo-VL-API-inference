import os
import torch
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, Field
from typing import List, Optional

# qwen_vl_utils is based on 'qwen-vl-utils[decord]==0.0.8'
# 详见 requirements.txt.  more details refer to requirements.txt
try:
    from qwen_vl_utils import process_vision_info
    from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError as e:
    print(f"关键模块导入失败: {e}")
    print("请确保 modelscope, qwen_vl_utils 等已正确安装并且在 Python 路径中。")
    raise

# --- 配置 ---
# 请确保这是您的本地模型路径。local model_path is based on 'MiMo-VL-7B-RL'
MODEL_PATH = "/hy-tmp/data/MiMo-VL-7B-RL"
# 临时文件存储目录 (用于上传的文件)。temp_upload_dir is set to 'uploaded-images'
TEMP_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded-images")
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

DEFAULT_PROMPT = "Describe this image." # 默认提示词。default prompt is set to 'Describe this image.'

# --- 全局变量 ---
app = FastAPI(title="MiMo VLM Image Description API")
model = None
processor = None

# --- Pydantic 模型定义 ---
class ImageUrlPayload(BaseModel):
    image_url: str
    prompt_text: Optional[str] = Field(default=DEFAULT_PROMPT, description="The prompt text to use with the image.")

class DescribeResponse(BaseModel):
    description: str
    prompt_used: str
    error: str = None

# --- 服务启动事件 ---
@app.on_event("startup")
async def startup_event():
    global model, processor
    print("服务启动中，开始加载模型和处理器...")
    if not os.path.isdir(MODEL_PATH):
        print(f"错误：模型路径 '{MODEL_PATH}' 不是一个有效的目录。")
        # 在FastAPI启动事件中，直接raise会导致服务无法启动，这里可以考虑记录日志并让服务以某种错误状态启动或不加载模型
        # 为了简单起见，这里我们仅打印错误，后续的API调用会因为模型未加载而失败
        return

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16, # 根据您的硬件和测试结果调整
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        print("模型和处理器加载成功。")
    except Exception as e:
        print(f"加载模型或处理器时发生严重错误: {str(e)}")
        # 同上，这里不直接 raise，让后续的 API 调用处理模型未加载的情况

# --- 核心处理逻辑 ---
async def get_image_description(image_path_or_url: str, prompt: str) -> str:
    if not model or not processor:
        raise HTTPException(status_code=503, detail="模型服务不可用，模型或处理器未成功加载。")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path_or_url},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 注意：这里我们需要原始的 input_ids 用于后续的修剪
        # processor(...) 返回的是一个字典，其中包含了 input_ids
        model_inputs = processor( # 将结果赋给 model_inputs
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        target_device = next(model.parameters()).device
        
        # 将 model_inputs 中的所有张量都移动到目标设备
        processed_inputs_on_device = {
            key: val.to(target_device) if hasattr(val, 'to') else val 
            for key, val in model_inputs.items()
        }


        # 使用处理并移动到设备后的输入
        generated_ids = model.generate(**processed_inputs_on_device, max_new_tokens=128)
        
        # --- 新增/修改的解码逻辑 ---
        # 获取原始输入的 token IDs，用于修剪
        # 注意: Hugging Face processor通常将input_ids放在返回字典的 'input_ids'键下
        input_ids_for_trimming = model_inputs['input_ids']

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_for_trimming, generated_ids)
        ]
        output_text_list = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # --- 结束新增/修改的解码逻辑 ---

        return output_text_list[0] if output_text_list else "无法生成描述。"

    except Exception as e:
        print(f"图像描述过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图像描述失败: {str(e)}")

# --- API 端点，仅支持url图片 ---
@app.post("/describe_url/", response_model=DescribeResponse)
async def describe_image_from_url(payload: ImageUrlPayload):
    effective_prompt = payload.prompt_text if payload.prompt_text is not None else DEFAULT_PROMPT
    print(f"收到URL请求: {payload.image_url}, 提示词: '{effective_prompt}'")
    try:
        description = await get_image_description(payload.image_url, effective_prompt)
        return DescribeResponse(description=description, prompt_used=effective_prompt)
    except HTTPException as http_exc: # 捕获由 get_image_description 抛出的 HTTPException
        return DescribeResponse(description="", prompt_used=effective_prompt, error=http_exc.detail)
    except Exception as e: # 捕获其他意外错误
        print(f"处理URL请求时发生意外错误: {str(e)}")
        return DescribeResponse(description="", prompt_used=effective_prompt, error=f"服务器内部错误: {str(e)}")

# --- API 支持字节上传文件 端点2 ---
@app.post("/describe_upload/", response_model=DescribeResponse)
async def describe_image_from_upload(
    image: UploadFile = File(...),
    prompt_text: Optional[str] = Form(default=DEFAULT_PROMPT, description="The prompt text to use with the image.")
):
    effective_prompt = prompt_text if prompt_text is not None else DEFAULT_PROMPT
    # 生成唯一文件名以避免冲突
    temp_filename = f"{uuid.uuid4()}_{image.filename}"
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
    
    print(f"收到文件上传请求: {image.filename}, 提示词: '{effective_prompt}', 将保存到: {temp_file_path}")

    try:
        # 保存上传的文件到临时位置
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        description = await get_image_description(temp_file_path, effective_prompt)
        return DescribeResponse(description=description, prompt_used=effective_prompt)
    except HTTPException as http_exc:
        return DescribeResponse(description="", prompt_used=effective_prompt, error=http_exc.detail)
    except Exception as e:
        print(f"处理文件上传时发生意外错误: {str(e)}")
        return DescribeResponse(description="", prompt_used=effective_prompt, error=f"服务器内部错误: {str(e)}")
    finally:
        # # 清理上传图片文件
        # if os.path.exists(temp_file_path):
        #     try:
        #         os.remove(temp_file_path)
        #         print(f"已删除临时文件: {temp_file_path}")
        #     except Exception as e_del:
        #         print(f"删除临时文件 {temp_file_path} 失败: {e_del}")
        await image.close()


# --- 运行服务器 (用于本地测试) ---
if __name__ == "__main__":
    import uvicorn
    # 确保 qwen_vl_utils.py 在PYTHONPATH中或与此脚本同目录
    # 检查 MODEL_PATH 是否正确
    if not os.path.isdir(MODEL_PATH):
       print(f"致命错误：模型路径 '{MODEL_PATH}' 不存在或不是一个目录。请在启动前配置正确的路径。")
       exit(1)
    
    print(f"启动 Uvicorn 服务器，监听 http://0.0.0.0:8000")
    print(f"模型路径配置为: {MODEL_PATH}")
    print(f"临时文件上传目录: {TEMP_UPLOAD_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)