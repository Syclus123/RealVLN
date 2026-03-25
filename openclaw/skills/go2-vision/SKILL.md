---
name: go2_vision
description: Capture images from Go2 robot dog's front camera and send to user. Use when user asks "看看前面有什么", "拍张照片", "前面有什么", "robot camera", etc.
---

# Go2 Vision - Robot Dog Camera

Capture images from Unitree Go2's front camera and send to user.

## Quick Start

```bash
# Capture photo (saved to ~/.openclaw/media/go2_camera.jpg)
python3 scripts/capture.py
```

## Workflow

### 1. Capture Photo
```bash
python3 /home/unitree/openclaw/skills/go2-vision/scripts/capture.py
```

Image saved to: `~/.openclaw/media/go2_camera.jpg`

**Note**: This path is in OpenClaw's media directory for Feishu compatibility.

### 2. Send Photo to User
**Use `message` tool with `media` parameter:**
```yaml
message:
  action: send
  media: "~/.openclaw/media/go2_camera.jpg"
```

**⚠️ Important:** 
- `message` + `media` = Send image to user ✅
- `image` tool = Analyze image with VLM (does NOT send to user) ❌

### 3. (Optional) Analyze Image Content
If user asked "看看前面有什么", use `image` tool to analyze:
```yaml
image:
  image: "~/.openclaw/media/go2_camera.jpg"
  prompt: "描述图片内容"
```

## Feishu-Specific Notes

### Media Path Requirements
Feishu plugin has strict path restrictions. Allowed paths:
- `~/.openclaw/media/` ✅ (recommended)
- Workspace-relative paths ✅
- `/tmp/` ❌ (blocked)
- `/home/unitree/` ❌ (blocked)

### Internal API Flow
The `message` tool with `media` automatically handles:
1. **Get Token**: `POST /auth/v3/tenant_access_token/internal`
2. **Upload Image**: `POST /im/v1/images` 
   - Form: `image_type=message`, `image=@file`
   - Returns: `image_key`
3. **Send Message**: `POST /im/v1/messages?receive_id_type=open_id`
   - Body: `{"receive_id": "...", "msg_type": "image", "content": "{\"image_key\": \"...\"}"}`

### Image Constraints
| Property | Limit |
|----------|-------|
| Max Size | 10MB |
| Formats | JPEG, PNG, WEBP, GIF, TIFF, BMP, ICO |

See tutorial: https://zhuanlan.zhihu.com/p/201049470403532950

## Complete Examples

**User**: "拍个照"
```yaml
# 1. Capture
exec: python3 scripts/capture.py

# 2. Send photo to user
message:
  action: send
  media: "~/.openclaw/media/go2_camera.jpg"
```

**User**: "看看前面有什么"
```yaml
# 1. Capture
exec: python3 scripts/capture.py

# 2. Send photo
message:
  action: send
  media: "~/.openclaw/media/go2_camera.jpg"

# 3. Analyze with VLM
image:
  image: "~/.openclaw/media/go2_camera.jpg"
  prompt: "你是一个性能优秀的VLM模型，描述图片内容"

# 4. Reply with description
```

## File Structure

```
~/.openclaw/media/
└── go2_camera.jpg          # Captured image (auto-saved here)

~/openclaw/skills/go2-vision/
├── SKILL.md                # This file
└── scripts/
    ├── capture.py          # Camera capture script
    └── send_image.py       # Standalone send helper (optional)
```

## References

- Feishu Upload Image: https://open.feishu.cn/document/server-docs/im-v1/image/create
- Feishu Send Message: https://open.feishu.cn/document/server-docs/im-v1/message/create
- Tutorial: https://zhuanlan.zhihu.com/p/201049470403532950
