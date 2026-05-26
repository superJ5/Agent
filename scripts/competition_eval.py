"""
比赛评测脚本 - 批量测试智能体并生成提交文件

功能:
    1. 读取测试问题 CSV（含问题、图片路径、会话ID）
    2. 逐条调用本地 /chat API 获取答案
    3. 生成比赛要求的 submission.csv 提交文件

用法:
    # 方式一：一键全流程（初始化 → 启动 → 测试 → 输出）
    python scripts/competition_eval.py --pipeline

    # 方式二：分步执行

    # 第一步：环境初始化（安装依赖 + 启动 Milvus + 入库知识库）
    python scripts/competition_eval.py --init

    # 第二步：启动服务
    python scripts/competition_eval.py --start

    # 第三步：批量测试（传入问题 CSV，输出答案 CSV）
    python scripts/competition_eval.py --test --input questions.csv --output submission.csv

    # 第四步：停止服务
    python scripts/competition_eval.py --stop

参数:
    --input     测试问题 CSV 路径（默认: data/test_questions.csv）
    --output    提交答案 CSV 路径（默认: data/submission.csv）
    --api-url   API 地址（默认: http://localhost:9900/chat）
    --token     Bearer Token（默认: 从 .env 读取）
    --workers   并发数（默认: 1，设为 >1 启用并发加速）
    --timeout   单题超时秒数，文本 20s / 多模态 30s（默认: 30）
    --skip-init 跳过初始化检查，直接测试

运行指令参考：
python3 scripts/competition_eval.py --test --input data/question_public.csv --output data/submission.csv --workers 8
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ── 项目根目录 ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 默认路径 ───────────────────────────────────────────────────────────────
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "test_questions.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "submission.csv"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

# ── API 配置 ───────────────────────────────────────────────────────────────
DEFAULT_API_URL = "http://localhost:9900/chat"
API_TIMEOUT_TEXT = 20       # 纯文本请求超时（秒）
API_TIMEOUT_MULTIMODAL = 30  # 多模态请求超时（秒）


# ══════════════════════════════════════════════════════════════════════════
# 第一步: 环境初始化
# ══════════════════════════════════════════════════════════════════════════

def cmd_init():
    """初始化开发环境：安装依赖 + 启动 Milvus + 入库知识库"""
    print("=" * 60)
    print("📦 [1/4] 检查 Python 虚拟环境...")
    print("=" * 60)

    venv_python = _get_venv_python()
    if not venv_python:
        print("⚠️   未检测到虚拟环境，是否创建？(y/n): ", end="")
        answer = input().strip().lower()
        if answer == "y":
            _run_cmd([sys.executable, "-m", "venv", ".venv"])
            print("✅ 虚拟环境已创建")
            venv_python = _get_venv_python()
        else:
            venv_python = sys.executable
            print("ℹ️   使用系统 Python:", venv_python)

    print("\n" + "=" * 60)
    print("📦 [2/4] 安装项目依赖...")
    print("=" * 60)

    # 优先使用 uv（更快）
    uv_path = _which("uv")
    if uv_path:
        _run_cmd([uv_path, "pip", "install", "-e", "."], cwd=PROJECT_ROOT)
    else:
        _run_cmd([venv_python, "-m", "pip", "install", "-e", "."], cwd=PROJECT_ROOT)

    print("\n" + "=" * 60)
    print("🐳 [3/4] 启动 Milvus 向量数据库...")
    print("=" * 60)

    # 检查 Docker 是否可用
    docker_path = _which("docker")
    if not docker_path:
        print("❌ 未检测到 Docker，请先安装 Docker Desktop")
        print("   下载地址: https://www.docker.com/products/docker-desktop/")
        sys.exit(1)

    # 检查 Docker 是否在运行
    result = _run_cmd([docker_path, "ps"], capture=True, check=False)
    if result.returncode != 0:
        print("⚠️   Docker 未运行，请先启动 Docker Desktop")
        sys.exit(1)

    # 启动 Milvus
    compose_file = PROJECT_ROOT / "vector-database.yml"
    _run_cmd([docker_path, "compose", "-f", str(compose_file), "up", "-d"], cwd=PROJECT_ROOT)
    print("⏳ 等待 Milvus 就绪（约 10 秒）...")
    time.sleep(10)

    print("\n" + "=" * 60)
    print("📚 [4/4] 入库手册知识库...")
    print("=" * 60)

    index_script = PROJECT_ROOT / "scripts" / "index_manual_chunks.py"
    if index_script.exists():
        _run_cmd([venv_python, str(index_script)])
    else:
        print("⚠️   未找到入库脚本，跳过知识库入库")
        print("   预期路径:", index_script)

    print("\n" + "=" * 60)
    print("✅ 环境初始化完成！")
    print("=" * 60)
    print("\n下一步: 启动服务")
    print(f"  python scripts/competition_eval.py --start")
    print("\n或者一键全流程:")
    print(f"  python scripts/competition_eval.py --pipeline")


# ══════════════════════════════════════════════════════════════════════════
# 第二步: 启动服务
# ══════════════════════════════════════════════════════════════════════════

def cmd_start():
    """启动 FastAPI 服务（后台运行）"""
    print("=" * 60)
    print("🚀 启动 FastAPI 服务...")
    print("=" * 60)

    # 创建日志目录
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DEFAULT_LOG_DIR / "server.log"
    pid_file = DEFAULT_LOG_DIR / "server.pid"

    # 检查是否已在运行
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if _is_process_running(old_pid):
                print(f"⚠️   服务已在运行 (PID: {old_pid})")
                print(f"    API: http://localhost:9900")
                print(f"    文档: http://localhost:9900/docs")
                return
        except (ValueError, OSError):
            pass

    venv_python = _get_venv_python()

    # Windows 使用 START /B 后台运行
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        proc = subprocess.Popen(
            [venv_python, "-m", "uvicorn", "app.main:app",
             "--host", "0.0.0.0", "--port", "9900"],
            cwd=PROJECT_ROOT,
            stdout=open(log_file, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            [venv_python, "-m", "uvicorn", "app.main:app",
             "--host", "0.0.0.0", "--port", "9900"],
            cwd=PROJECT_ROOT,
            stdout=open(log_file, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )

    # 保存 PID
    pid_file.write_text(str(proc.pid))

    print(f"⏳ 等待服务就绪...")
    time.sleep(3)

    # 等待服务启动（最多等待 15 秒）
    import urllib.request
    import urllib.error

    for i in range(15):
        try:
            resp = urllib.request.urlopen("http://localhost:9900/health", timeout=3)
            if resp.status == 200:
                print(f"✅ FastAPI 服务启动成功 (PID: {proc.pid})")
                print(f"   🌐 API:     http://localhost:9900")
                print(f"   📚 文档:    http://localhost:9900/docs")
                print(f"   📝 日志:    {log_file}")
                return
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(1)

    print("❌ 服务启动超时，请检查日志:", log_file)
    print("   手动启动: python -m uvicorn app.main:app --host 0.0.0.0 --port 9900")


# ══════════════════════════════════════════════════════════════════════════
# 第三步: 批量测试
# ══════════════════════════════════════════════════════════════════════════

async def cmd_test(args):
    """读取测试 CSV → 调用 API → 生成提交 CSV"""
    input_csv = Path(args.input)
    output_csv = Path(args.output)
    api_url = args.api_url
    token = args.token or _load_token_from_env()
    max_workers = args.workers
    timeout = args.timeout

    # 读取测试问题
    if not input_csv.exists():
        print(f"❌ 测试文件不存在: {input_csv}")
        print(f"   请指定正确的 --input 路径")
        sys.exit(1)

    test_cases = _load_test_cases(input_csv)
    print(f"\n📋 共加载 {len(test_cases)} 条测试问题")
    print(f"🔗 API: {api_url}")
    print(f"⚡ 并发: {max_workers}")
    print(f"⏱️  超时: {timeout}s")
    print()

    # 执行测试
    results = await _run_batch_test(
        test_cases=test_cases,
        api_url=api_url,
        token=token,
        max_workers=max_workers,
        timeout=timeout,
    )

    # 写入输出 CSV
    _write_submission_csv(results, output_csv)

    # 打印统计
    success_count = sum(1 for r in results if r.get("success"))
    fail_count = sum(1 for r in results if not r.get("success"))
    total_time = sum(r.get("elapsed", 0) for r in results)

    print("\n" + "=" * 60)
    print("📊 测试统计")
    print("=" * 60)
    print(f"   总题数:    {len(results)}")
    print(f"   ✅ 成功:   {success_count}")
    print(f"   ❌ 失败:   {fail_count}")
    print(f"   ⏱️  总耗时: {total_time:.1f}s")
    print(f"   ⚡ 平均:   {total_time / max(len(results), 1):.2f}s/题")
    print(f"\n📄 提交文件已生成: {output_csv}")
    print(f"   可上传至比赛评分系统")


def _load_test_cases(csv_path: Path) -> list[dict]:
    """
    从 CSV 加载测试问题。

    CSV 格式约定（与比赛对齐）:
        id,question,images,session_id
        1,我的电钻指示灯闪烁代表什么？,,
        2,健身追踪器表带尺寸？,"images/photo1.jpg;images/photo2.png",kf_session_001
        3,物流一直待揽收？,,

    参数说明:
        id          - 问题编号（必填，对应提交文件中的 id）
        question    - 问题文本（必填）
        images      - 图片路径列表（可选，多个用分号 ; 分隔）
        session_id  - 会话ID（可选，为空则自动生成）

    Args:
        csv_path: CSV 文件路径

    Returns:
        list[dict]: 测试用例列表
    """
    test_cases = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # 去除列名空白
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        for row_num, row in enumerate(reader, start=2):
            # 解析问题编号
            raw_id = row.get("id", "").strip()
            if not raw_id:
                print(f"⚠️   第 {row_num} 行缺少 id，跳过")
                continue

            try:
                question_id = int(raw_id)
            except ValueError:
                print(f"⚠️   第 {row_num} 行 id 非法 ('{raw_id}')，跳过")
                continue

            # 解析问题文本
            question = row.get("question", "").strip()
            if not question:
                print(f"⚠️   第 {row_num} 行 question 为空，跳过")
                continue

            # 解析图片路径（可选）
            images: list[str] = []
            raw_images = row.get("images", "").strip()
            if raw_images:
                for img_path in raw_images.split(";"):
                    img_path = img_path.strip()
                    if img_path:
                        # 尝试读取文件并转为 Base64
                        full_path = Path(img_path)
                        if not full_path.is_absolute():
                            full_path = PROJECT_ROOT / img_path

                        if full_path.exists():
                            b64 = _image_to_base64(full_path)
                            images.append(b64)
                            print(f"   📷 加载图片: {full_path}")
                        else:
                            print(f"   ⚠️  图片不存在: {full_path}")

            # 解析会话 ID（可选）
            session_id = row.get("session_id", "").strip()
            if not session_id:
                session_id = f"kf_session_{uuid.uuid4().hex}"

            test_cases.append({
                "id": question_id,
                "question": question,
                "images": images,
                "session_id": session_id,
            })

    # 按 id 排序
    test_cases.sort(key=lambda x: x["id"])
    return test_cases


def _image_to_base64(image_path: Path) -> str:
    """
    将图片文件转为 Base64 格式。

    Args:
        image_path: 图片文件路径

    Returns:
        str: data:image/{format};base64,{encoded_string}
    """
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/png")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{encoded}"


async def _run_single_test(
    case: dict,
    api_url: str,
    token: str,
    timeout: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    调用 /chat API 测试单条问题。

    Args:
        case: 测试用例
        api_url: API 地址
        token: Bearer Token
        timeout: 超时秒数
        semaphore: 并发信号量

    Returns:
        dict: 测试结果
    """
    import httpx

    async with semaphore:
        start_time = time.time()
        question_id = case["id"]
        question = case["question"]
        images = case["images"]
        session_id = case["session_id"]

        # 判断是否为多模态请求，选择合适的超时
        effective_timeout = timeout
        if images:
            effective_timeout = max(timeout, API_TIMEOUT_MULTIMODAL)

        # 构建请求体
        payload: dict[str, Any] = {
            "question": question,
            "session_id": session_id,
        }
        if images:
            payload["images"] = images

        try:
            async with httpx.AsyncClient(timeout=effective_timeout) as client:
                headers = {
                    "Content-Type": "application/json",
                }
                if token:
                    headers["Authorization"] = f"Bearer {token}"

                resp = await client.post(api_url, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()

                # 解析比赛标准响应
                code = body.get("code", -1)
                if code == 0:
                    answer = body.get("data", {}).get("answer", "")
                    elapsed = time.time() - start_time
                    print(f"   ✅ [ID={question_id}] {elapsed:.1f}s | {question[:40]}...")
                    return {
                        "id": question_id,
                        "ret": answer,
                        "success": True,
                        "elapsed": elapsed,
                        "session_id": session_id,
                    }
                else:
                    error_msg = body.get("msg", "未知错误")
                    elapsed = time.time() - start_time
                    print(f"   ❌ [ID={question_id}] API 返回错误: {error_msg}")
                    return {
                        "id": question_id,
                        "ret": f"ERROR: {error_msg}",
                        "success": False,
                        "elapsed": elapsed,
                        "session_id": session_id,
                    }

        except httpx.TimeoutException:
            elapsed = time.time() - start_time
            print(f"   ❌ [ID={question_id}] 请求超时 ({elapsed:.1f}s)")
            return {
                "id": question_id,
                "ret": "ERROR: 请求超时",
                "success": False,
                "elapsed": elapsed,
                "session_id": session_id,
            }

        except Exception as exc:
            elapsed = time.time() - start_time
            print(f"   ❌ [ID={question_id}] 请求失败: {exc}")
            return {
                "id": question_id,
                "ret": f"ERROR: {exc}",
                "success": False,
                "elapsed": elapsed,
                "session_id": session_id,
            }


async def _run_batch_test(
    test_cases: list[dict],
    api_url: str,
    token: str,
    max_workers: int,
    timeout: int,
) -> list[dict]:
    """
    批量运行测试用例。

    Args:
        test_cases: 测试用例列表
        api_url: API 地址
        token: Bearer Token
        max_workers: 最大并发数
        timeout: 超时秒数

    Returns:
        list[dict]: 测试结果列表
    """
    print("=" * 60)
    print("🧪 开始批量测试...")
    print("=" * 60)
    print()

    semaphore = asyncio.Semaphore(max_workers)

    tasks = [
        _run_single_test(case, api_url, token, timeout, semaphore)
        for case in test_cases
    ]

    results = await asyncio.gather(*tasks)

    # 按 id 排序
    results.sort(key=lambda x: x["id"])
    return results


def _write_submission_csv(results: list[dict], output_path: Path):
    """
    写入比赛提交文件 submission.csv。

    格式:
        id,ret
        1,回答内容...
        2,回答内容...

    Args:
        results: 测试结果列表
        output_path: 输出 CSV 路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "ret"])
        for r in results:
            writer.writerow([r["id"], r["ret"]])

    print(f"📄 提交文件已保存: {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# 第四步: 停止服务
# ══════════════════════════════════════════════════════════════════════════

def cmd_stop():
    """停止 FastAPI 服务"""
    print("=" * 60)
    print("🛑 停止 FastAPI 服务...")
    print("=" * 60)

    pid_file = DEFAULT_LOG_DIR / "server.pid"
    if not pid_file.exists():
        print("⚠️   未找到运行中的服务（PID 文件不存在）")
        return

    try:
        pid = int(pid_file.read_text().strip())
        if sys.platform == "win32":
            _run_cmd(["taskkill", "/F", "/PID", str(pid)], check=False)
        else:
            _run_cmd(["kill", str(pid)], check=False)

        pid_file.unlink(missing_ok=True)
        print(f"✅ 服务已停止 (PID: {pid})")

    except (ValueError, OSError) as e:
        print(f"⚠️   停止服务失败: {e}")
        pid_file.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 一键全流程
# ══════════════════════════════════════════════════════════════════════════

def cmd_pipeline(args):
    """一键全流程：初始化 → 启动 → 测试 → 停止"""
    # 第一步: 初始化
    cmd_init()

    # 第二步: 启动
    cmd_start()

    # 等待服务完全就绪
    print("\n⏳ 等待服务稳定...")
    time.sleep(3)

    # 第三步: 测试
    asyncio.run(cmd_test(args))

    # 询问是否停止
    print("\n" + "=" * 60)
    answer = input("🛑 是否停止服务？(Y/n): ").strip().lower()
    if answer != "n":
        cmd_stop()


# ══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════

def _get_venv_python() -> str:
    """获取虚拟环境的 Python 路径"""
    if sys.platform == "win32":
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

    if venv_python.exists():
        return str(venv_python)

    # 回退到当前 Python
    return sys.executable


def _load_token_from_env() -> str:
    """从 .env 文件或环境变量加载 API_BEARER_TOKEN"""
    # 先从环境变量读取
    token = os.environ.get("API_BEARER_TOKEN", "")

    # 再从 .env 文件读取
    if not token:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("API_BEARER_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("\"'")
                    break

    return token


def _run_cmd(cmd: list[str], cwd: Path | None = None, capture: bool = False,
             check: bool = True) -> subprocess.CompletedProcess:
    """运行命令并打印输出"""
    print(f"   $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        capture_output=capture,
        text=True,
        encoding="utf-8",
    )
    if capture:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    if check and result.returncode != 0:
        print(f"   ⚠️  命令退出码: {result.returncode}")

    return result


def _which(name: str) -> str | None:
    """查找可执行文件路径"""
    return os.pathsep.join(
        p for p in (
            _find_on_path(name)
            or [_get_venv_python().rsplit("\\" if sys.platform == "win32" else "/", 1)[0]]
        )
        if p
    ) or None


def _find_on_path(name: str) -> list[str]:
    """在 PATH 中查找可执行文件"""
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for p in paths:
        full = os.path.join(p, name)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return [full]
        # Windows 下尝试加 .exe
        if sys.platform == "win32":
            full_exe = full + ".exe"
            if os.path.isfile(full_exe):
                return [full_exe]
    return []


def _is_process_running(pid: int) -> bool:
    """检查进程是否在运行"""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, encoding="utf-8",
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.SubprocessError):
        return False


# ══════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🏆 多模态客服智能体 - 比赛评测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 一键全流程
  python scripts/competition_eval.py --pipeline

  # 分步执行
  python scripts/competition_eval.py --init
  python scripts/competition_eval.py --start
  python scripts/competition_eval.py --test --input data/test_questions.csv --output data/submission.csv
  python scripts/competition_eval.py --stop

  # 并发加速（8 线程）
  python scripts/competition_eval.py --test --workers 8

  # 指定 API 地址
  python scripts/competition_eval.py --test --api-url http://192.168.1.100:9900/chat
        """,
    )

    # 操作模式
    parser.add_argument("--pipeline", action="store_true", help="一键全流程（初始化→启动→测试→停止）")
    parser.add_argument("--init", action="store_true", help="初始化环境（安装依赖+启动Milvus+入库知识库）")
    parser.add_argument("--start", action="store_true", help="启动 FastAPI 服务")
    parser.add_argument("--test", action="store_true", help="批量测试（读取 CSV → 调用 API → 输出 CSV）")
    parser.add_argument("--stop", action="store_true", help="停止 FastAPI 服务")

    # 测试参数
    parser.add_argument("--input", default=str(DEFAULT_INPUT_CSV), help=f"测试问题 CSV 路径（默认: {DEFAULT_INPUT_CSV}）")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_CSV), help=f"提交答案 CSV 路径（默认: {DEFAULT_OUTPUT_CSV}）")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API 地址（默认: {DEFAULT_API_URL}）")
    parser.add_argument("--token", default="sk-7102e3de117945928d99e398e6db3cfa", help="Bearer Token（默认）")
    parser.add_argument("--workers", type=int, default=8, help="并发数（默认: 1）")
    parser.add_argument("--timeout", type=int, default=30, help="单题超时秒数（默认: 30）")

    args = parser.parse_args()

    # 确定操作模式
    mode = None
    for flag in ["pipeline", "init", "start", "test", "stop"]:
        if getattr(args, flag):
            mode = flag
            break

    if not mode:
        parser.print_help()
        print("\n⚠️   请指定操作模式，例如: --test")
        print("   使用 --pipeline 可一键全流程执行")
        return

    print(f"🏆 多模态客服智能体 - 比赛评测工具")
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print()

    if mode == "pipeline":
        cmd_pipeline(args)
    elif mode == "init":
        cmd_init()
    elif mode == "start":
        cmd_start()
    elif mode == "test":
        asyncio.run(cmd_test(args))
    elif mode == "stop":
        cmd_stop()


if __name__ == "__main__":
    main()
