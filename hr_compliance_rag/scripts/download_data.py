"""
数据下载脚本 - 从 git-lfs 媒体直链下载真实法规数据集

数据源: twang2218/law-datasets (爬自国家法律法规数据库 flk.npc.gov.cn)
产物:   data/raw/laws.json
"""

import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

# git-lfs 媒体直链(raw.githubusercontent 只会返回 LFS 指针文件)
LFS_URL = (
    "https://media.githubusercontent.com/media/twang2218/law-datasets/"
    "main/law-and-regulations/laws.json.zip"
)

# 项目根目录(scripts 的上一级)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "laws.json.zip"
JSON_PATH = RAW_DIR / "laws.json"

MAX_RETRY = 3


def download(url: str, dest: Path) -> None:
    """带重试的下载"""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(f"[{attempt}/{MAX_RETRY}] 正在下载 {url}")
            print(f"        -> {dest} (~98MB，请耐心等待)")
            # 流式下载并显示进度
            with urllib.request.urlopen(url, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            sys.stdout.write(f"\r        进度: {pct}% ({downloaded}/{total} bytes)")
                            sys.stdout.flush()
            print("\n下载完成")
            return
        except Exception as e:
            print(f"\n下载失败: {e}")
            if attempt < MAX_RETRY:
                wait = 2 * attempt
                print(f"{wait}s 后重试...")
                time.sleep(wait)
            else:
                raise RuntimeError("多次重试后仍下载失败，请检查网络或手动下载") from e


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if JSON_PATH.exists():
        print(f"已存在 {JSON_PATH}，跳过下载。")
        return

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 102396322:
        download(LFS_URL, ZIP_PATH)

    print(f"正在解压 {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        print(f"压缩包内文件: {names}")
        zf.extractall(RAW_DIR)

    if JSON_PATH.exists():
        size_mb = JSON_PATH.stat().st_size / (1024 * 1024)
        print(f"[OK] 解压完成: {JSON_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"[WARN] 解压完成，但未找到预期的 laws.json，请检查 {RAW_DIR} 内文件")


if __name__ == "__main__":
    main()
