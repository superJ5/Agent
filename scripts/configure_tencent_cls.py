"""交互式写入腾讯云 CLS 配置，避免把 SecretKey 发到聊天或写入 shell 历史。"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

from dotenv import set_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env.local"


def require_value(prompt: str, *, secret: bool = False) -> str:
    while True:
        value = (getpass.getpass(prompt) if secret else input(prompt)).strip()
        if value:
            return value
        print("该项不能为空，请重新输入。")


def main() -> None:
    secret_id = require_value("请输入腾讯云 SecretId: ")
    secret_key = require_value("请输入腾讯云 SecretKey（输入不可见）: ", secret=True)

    values = {
        "CLS_DATA_SOURCE": "tencent",
        "TENCENTCLOUD_SECRET_ID": secret_id,
        "TENCENTCLOUD_SECRET_KEY": secret_key,
        "TENCENT_CLS_REGION": "ap-guangzhou",
        "TENCENT_CLS_TOPIC_ID": "5881f767-15e0-4285-9e9f-3a5bed015ae8",
        "TENCENT_CLS_TOPIC_NAME": "super-biz-agent-app",
        "TENCENT_CLS_SERVICE_NAME": "superj-vm",
    }
    ENV_FILE.touch(exist_ok=True)
    for key, value in values.items():
        set_key(str(ENV_FILE), key, value, quote_mode="always")
    os.chmod(ENV_FILE, 0o600)

    print("腾讯云 CLS 配置已写入 .env.local，密钥内容未打印。")


if __name__ == "__main__":
    main()
