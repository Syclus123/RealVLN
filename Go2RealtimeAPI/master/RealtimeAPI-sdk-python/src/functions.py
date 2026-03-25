"""function calling lists.

四大类:
  - control:    机器狗动作控制
  - chat:       闲聊对话
  - vqa:     图像问答
  - nav:    导航
"""

# ══════════════════════════════════════════════
# 意图类型
# ══════════════════════════════════════════════

INTENT_CONTROL = "control"
INTENT_CHAT = "chat"
INTENT_VQA = "vqa"
INTENT_NAV = "navigation"

ACTION_TOOLS = [
    # ==================== 姿态 ====================
    {
        "type": "function",
        "function": {
            "name": "stand_up",
            "description": "站立起来",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stand_down",
            "description": "趴下/坐下",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "balance",
            "description": "平衡站立",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recovery",
            "description": "恢复站立(摔倒后起立)",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "damp",
            "description": "阻尼模式/关节放松",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "停止移动/停下来",
        }
    },

    # ==================== 移动 ====================
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "向前走",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "number",
                        "description": "前进速度，范围 0.0~1.0，默认 0.3"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_backward",
            "description": "后退",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "number",
                        "description": "后退速度，范围 0.0~1.0，默认 0.3"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_lateral",
            "description": "横向移动/左平移/右平移",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "number",
                        "description": "横向移动速度，范围 0.0~0.5，默认 0.3"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_left",
            "description": "左转",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "number",
                        "description": "左转速度，范围 0.0~1.0，默认 0.5"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_right",
            "description": "右转",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "number",
                        "description": "右转速度，范围 0.0~1.0，默认 0.5"
                    }
                },
                "required": []
            }
        }
    },

    # ==================== 即时特技 ====================
    {
        "type": "function",
        "function": {
            "name": "left_flip",
            "description": "左空翻",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "back_flip",
            "description": "后空翻",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "free_walk",
            "description": "自由行走/随便走走",
        }
    },

    # ==================== 持续特技 ====================
    {
        "type": "function",
        "function": {
            "name": "handstand",
            "description": "倒立",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "持续时长(秒)，范围 1.0~10.0，默认 4.0"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "free_jump",
            "description": "跳跃",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "持续时长(秒)，范围 1.0~10.0，默认 4.0"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "free_bound",
            "description": "蹦跳",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "持续时长(秒)，范围 1.0~10.0，默认 2.0"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "free_avoid",
            "description": "自动避障行走",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "持续时长(秒)，范围 1.0~10.0，默认 2.0"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "walk_upright",
            "description": "直立行走",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "持续时长(秒)，范围 1.0~10.0，默认 4.0"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cross_step",
            "description": "交叉步",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "持续时长(秒)，范围 1.0~10.0，默认 4.0"
                    }
                },
                "required": []
            }
        }
    },

    # ==================== 社交/表情 ====================
    {
        "type": "function",
        "function": {
            "name": "sit",
            "description": "坐下",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hello",
            "description": "打招呼/挥手",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stretch",
            "description": "伸展/伸懒腰",
        }
    },
]

GO2_TOOLS = [
    # ==================== CHAT ====================
    {
        "type": "function",
        "function": {
            "name": "chat",
            "description": "日常闲聊、打招呼、或者回答不需要视觉和导航的通用知识问题时调用。",
        }   
    },
    
    # ==================== NAV ====================
    {
        "type": "function",
        "function": {
            "name": "nav",
            "description": "导航指令，到哪儿去，帮我找个东西，看看某个东西在哪儿，当用户要求去某地、找某物、询问某物在哪里时调用。",
        }
    },

    # ==================== VQA ====================
    {
        "type": "function",
        "function": {
            "name": "vqa",
            "description": "图像问答，看看这个是什么，看看这里有什么，当用户要求看看这是什么、描述当前场景、或者询问眼前物品的具体细节时调用。",
        }
    },
]

GO2_TOOLS.extend(ACTION_TOOLS)