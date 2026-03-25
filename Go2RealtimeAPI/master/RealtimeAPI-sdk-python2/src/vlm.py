import base64
import os
import dashscope 
from dashscope import MultiModalConversation
import asyncio

# 以下为北京地域base_url，若使用弗吉尼亚地域模型，需要将base_url换成 https://dashscope-us.aliyuncs.com/api/v1
# 若使用新加坡地域的模型，需将base_url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

#  编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class QwenVLM:
    def __init__(self, logger, api_key=os.getenv("DASHSCOPE_API_KEY"), model_name="qwen3.5-plus"):
        self.api_key = api_key
        self.model_name = model_name
        self.retry_num = 2
        self.logger = logger

    def start(self, messages, tools=None):
        retry_num = self.retry_num
        text = ""

        while retry_num > 0:
            try:
                response = MultiModalConversation.call(
                    api_key=self.api_key,
                    model=self.model_name,
                    messages=messages,
                    result_format="message",
                )
                if response.status_code == 200:
                    text = response.output.choices[0].message.content[0]["text"]
                    break
                else:
                    retry_num -= 1
                    # ✅ 建议：记录失败原因
                    self.logger.warning(
                        f"LLM 调用失败: status={response.status_code}, "
                        f"msg={getattr(response, 'message', '')}"
                    )

            except Exception as e:          # ✅ 不要裸 except:
                self.logger.error(f"LLM 调用异常: {e}")
                retry_num -= 1

        return text

async def call_vlm_api(
    logger,
    vlm: QwenVLM,
    user_image_path: str,
    user_text: str,
    conversation_history: list[dict],
    tools=None,
) -> tuple[str | None, object | None]:     # ✅ 返回类型修正为 tuple
    """调用 LLM，一次性返回完整回复."""
    # logger.info(f"[LLM] 输入: {user_text}")

    base64_image = encode_image(user_image_path)

    messages = []
    messages.extend(conversation_history)
    messages.append(
        {
        "role": "user", 
        "content": [            
            {"image": f"data:image/{os.path.splitext(user_image_path)[1]};base64,{base64_image}"},
            {"text": f"{user_text}"},
            ],
        }
        )

    # ✅ 核心：to_thread 不阻塞事件循环
    text = await asyncio.to_thread(vlm.start, messages, tools)
    return text

if __name__ == "__main__":
    import logging

    logger = logging.getLogger("vlm")
    logger.setLevel(logging.DEBUG)

    image_path = "/home/unitree/.openclaw/media/go2_camera.jpg"
    vlm = QwenVLM(logger)
    text = asyncio.run(call_vlm_api(logger, vlm, image_path, "图片里面有什么, 简单总结一下", []))
    print(text)


