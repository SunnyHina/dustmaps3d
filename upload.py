import os
import shutil
import subprocess
import sys

# === 用户可配置项 ===
PACKAGE_NAME = "dustmaps3d"
PYPI_REPO = "dustmaps3d"
# ====================

def run(cmd, cwd=None):
    """运行 shell 命令，自动退出错误"""
    print(f"📦 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ 命令失败，错误码 {result.returncode}")
        sys.exit(result.returncode)

def clean_previous_builds():
    """清理旧的构建文件和 __pycache__"""
    print("🧹 Cleaning previous build files...")
    for folder in ['build', 'dist', f'{PACKAGE_NAME}.egg-info']:
        shutil.rmtree(folder, ignore_errors=True)

    for root, dirs, _ in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                pycache_path = os.path.join(root, d)
                print(f"🗑️ Removing __pycache__: {pycache_path}")
                shutil.rmtree(pycache_path, ignore_errors=True)

def ensure_dependencies():
    """确保 build 和 twine 已安装"""
    print("🔍 Checking required packages...")
    missing = []
    try:
        import build
    except ImportError:
        missing.append("build")
    try:
        import twine
    except ImportError:
        missing.append("twine")

    if missing:
        print(f"📦 安装缺失依赖: {' '.join(missing)}")
        run(f"{sys.executable} -m pip install " + " ".join(missing))

def build_package():
    """构建 tar.gz 和 wheel 包"""
    print("🛠️ Building package...")
    run(f"{sys.executable} -m build")

def upload_to_pypi():
    """上传包到 PyPI 或 TestPyPI"""
    print("⬆️ Uploading to PyPI...")
    run(f"{sys.executable} -m twine upload --repository {PYPI_REPO} dist/*")

def main():
    ensure_dependencies()
    clean_previous_builds()
    build_package()
    upload_to_pypi()
    print("✅ 发布完成：已上传到 PyPI")

if __name__ == "__main__":
    main()
