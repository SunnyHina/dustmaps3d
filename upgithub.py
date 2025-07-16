import subprocess
import os
import sys
from pathlib import Path
from datetime import date

# ====== 用户可修改的设置 ======
TAG = "v2.1.0"  # GitHub Release 的标签
ASSET_PATH = Path("D:/_3d_map_data/data_v2.1.parquet")  # 要上传的数据文件路径
ASSET_NAME = "data_v2.1.parquet"  # 上传后在 release 中显示的文件名
REPO = "Grapeknight/dustmaps3d"  # GitHub 仓库名
RELEASE_TITLE = "Dustmaps3D v2.1.0"
RELEASE_NOTES = f"""
📦 Updated data release for Dustmaps3D

- 🔢 Version: {TAG}
- 📅 Date: {date.today()}
- 📁 File: `{ASSET_NAME}` (~350MB)

👉 If GitHub download fails due to network issues, you can get the data via:
🔗 NADC: https://nadc.china-vo.org/res/r101619/
"""

# ====== 工具函数 ======
def run(cmd: str, cwd: Path = None):
    print(f"📦 Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        if allow_error:
            print("⚠️ 命令失败，但继续执行...")
        else:
            print("❌ 命令执行失败！")
            sys.exit(1)

def check_git_config():
    """检查 git 用户信息是否设置"""
    print("🔍 检查 Git 用户配置...")
    name = subprocess.getoutput("git config user.name")
    email = subprocess.getoutput("git config user.email")

    if not name.strip() or not email.strip():
        print("⚠️ Git 用户信息未配置，正在设置...")
        run('git config --global user.name "Grapeknight"')
        run('git config --global user.email "wt@mail.bnu.edu.cn"')
    else:
        print(f"✅ Git 用户已设置为：{name} <{email}>")

def check_git_remote():
    """检查远程是否使用 SSH，建议更换为 SSH 提高稳定性"""
    print("🔍 检查 Git 远程地址...")
    remote = subprocess.getoutput("git remote get-url origin")
    if remote.startswith("https://"):
        print("⚠️ 当前远程使用 HTTPS，推荐改为 SSH 更稳定")
        print("👉 正在修改为 SSH...")
        run(f"git remote set-url origin git@github.com:{REPO}.git")
    else:
        print("✅ Git 远程地址已是 SSH")

def push_code_to_github():
    print("🚀 推送代码到 GitHub 仓库...")
    run("git add .")
    # 如果没有实际改动，跳过 commit 不报错
    run('git commit -m "🔄 Update version, docs, and data link" || echo \"✅ 无需提交\"')
    run("git push origin main")

# ====== 创建 release 并上传数据文件 ======
def upload_release_asset():
    """将数据文件上传至 GitHub Releases"""
    print("📤 上传数据文件到 GitHub Releases...")

    if not DATA_PATH.exists():
        print(f"❌ 文件未找到: {DATA_PATH}")
        sys.exit(1)

    # 创建 release（如果不存在则忽略错误）
    run(f'gh release create {RELEASE_TAG} "{DATA_PATH}" --title "{RELEASE_NAME}" --notes "{RELEASE_DESC}"', allow_error=True)

    # 或者追加上传
    run(f'gh release upload {RELEASE_TAG} "{DATA_PATH}" --clobber')

def main():
    check_git_config()
    check_git_remote()
    push_code_to_github()
    upload_release_asset()
    print("✅ 所有操作已完成！")

if __name__ == "__main__":
    main()
