import asyncio
import os

from typing import Optional, List

import dashscope
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

class QwenLLM:
    def __init__(self, logger, api_key=os.getenv("DASHSCOPE_API_KEY"), model_name="qwen3-30b-a3b-instruct-2507"):
        self.api_key = api_key
        self.model_name = model_name
        self.retry_num = 2
        self.logger = logger

    def start(self, messages, tools=None):
        retry_num = self.retry_num
        text, tool_call = None, None

        while retry_num > 0:
            try:
                response = dashscope.Generation.call(
                    api_key=self.api_key,
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    result_format="message",
                )
                if response.status_code == 200:
                    output_message = response.output.choices[0].message
                    if "tool_calls" not in output_message or output_message.tool_calls is None:
                        text = output_message.content
                    else:
                        tool_call = output_message.tool_calls[0]
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

        return text, tool_call


async def call_llm_api(
    logger,
    llm: QwenLLM,
    user_text: str,
    conversation_history: list[dict],
    system_prompt: str = "",
    tools=None,
) -> tuple[str | None, object | None]:     # ✅ 返回类型修正为 tuple
    """调用 LLM，一次性返回完整回复."""
    # logger.info(f"[LLM] 输入: {user_text}")

    messages = []
    if system_prompt:                       # ✅ 简化：空字符串本身就是 falsy
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    # ✅ 核心：to_thread 不阻塞事件循环
    text, tool_call = await asyncio.to_thread(llm.start, messages, tools)
    return text, tool_call