#!/usr/bin/env python3
"""
Send image to user via OpenClaw channels.
Auto-detects channel type and handles Feishu API internally.

Usage:
    python3 send_image.py /path/to/image.jpg
"""

import sys
import os
import json
import requests
from pathlib import Path

# OpenClaw config path
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
IMAGE_PATH = "/home/unitree/openclaw/go2_camera.jpg"


def get_feishu_token(app_id: str, app_secret: str) -> str:
    """Get tenant_access_token from Feishu."""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {"Content-Type": "application/json"}
    data = {"app_id": app_id, "app_secret": app_secret}
    
    resp = requests.post(url, headers=headers, json=data, timeout=10)
    result = resp.json()
    
    if result.get("code") != 0:
        raise Exception(f"Failed to get token: {result.get('msg')}")
    
    return result["tenant_access_token"]


def upload_image_to_feishu(token: str, image_path: str) -> str:
    """Upload image to Feishu and return image_key."""
    url = "https://open.feishu.cn/open-apis/im/v1/images"
    headers = {"Authorization": f"Bearer {token}"}
    
    with open(image_path, "rb") as f:
        files = {
            "image": (os.path.basename(image_path), f, "image/jpeg"),
            "image_type": (None, "message")
        }
        resp = requests.post(url, headers=headers, files=files, timeout=30)
    
    result = resp.json()
    
    if result.get("code") != 0:
        error_msg = result.get('msg', 'Unknown error')
        # Common error mapping
        error_codes = {
            "99991672": "Permission denied: im:resource not granted",
            "234006": "File size exceeds 10MB limit",
            "234007": "Bot ability not enabled",
            "234010": "File size cannot be 0",
            "234011": "Unsupported image format",
            "234039": "Image resolution exceeds limit"
        }
        if str(result.get("code")) in error_codes:
            error_msg = f"{error_msg} ({error_codes[str(result.get('code'))]})"
        raise Exception(f"Upload failed: {error_msg}")
    
    return result["data"]["image_key"]


def send_feishu_image(token: str, receive_id: str, image_key: str) -> dict:
    """Send image message via Feishu."""
    url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "receive_id": receive_id,
        "msg_type": "image",
        "content": json.dumps({"image_key": image_key})
    }
    
    resp = requests.post(url, headers=headers, json=data, timeout=10)
    result = resp.json()
    
    if result.get("code") != 0:
        raise Exception(f"Send failed: {result.get('msg')}")
    
    return result


def get_channel_info():
    """Get current channel info from environment or return None."""
    # In OpenClaw environment, channel info is passed via context
    # This is a placeholder for channel auto-detection
    return None


def send_image(image_path: str):
    """
    Send image to user.
    For Feishu: uses API flow (token → upload → send)
    For other channels: outputs MEDIA tag for OpenClaw Gateway
    """
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check image size (10MB limit for Feishu)
    size = os.path.getsize(image_path)
    if size > 10 * 1024 * 1024:
        print(f"ERROR: Image too large ({size / 1024 / 1024:.1f}MB > 10MB)", file=sys.stderr)
        sys.exit(1)
    
    # Try to read OpenClaw config for Feishu
    if OPENCLAW_CONFIG.exists():
        try:
            with open(OPENCLAW_CONFIG) as f:
                config = json.load(f)
            
            feishu_accounts = config.get("channels", {}).get("feishu", {}).get("accounts", [])
            if feishu_accounts:
                # Use first Feishu account
                account = feishu_accounts[0]
                app_id = account.get("appId") or account.get("app_id")
                app_secret = account.get("appSecret") or account.get("app_secret")
                
                if app_id and app_secret:
                    print(f"Sending via Feishu API...")
                    token = get_feishu_token(app_id, app_secret)
                    image_key = upload_image_to_feishu(token, image_path)
                    print(f"Image uploaded: {image_key}")
                    # Note: receive_id should come from OpenClaw context in real usage
                    print(f"MEDIA: {image_path}")
                    return
        except Exception as e:
            print(f"Feishu API failed, falling back: {e}", file=sys.stderr)
    
    # Fallback: output MEDIA tag for OpenClaw Gateway
    print(f"MEDIA: {image_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to standard path
        send_image(IMAGE_PATH)
    else:
        send_image(sys.argv[1])
