import os
import shutil
import subprocess

# === 用户可配置项 ===
PACKAGE_NAME = "dustmaps3d"
PYPI_REPO = "dustmaps3d"  # 自定义的 PyPI 仓库名称（在 .pypirc 中配置）
# ====================

def run(cmd, cwd=None):
    """运行 shell 命令"""
    print(f"📦 Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def clean_previous_builds():
    """清理旧的 dist、build、egg-info、__pycache__ 等目录"""
    print("🧹 Cleaning previous build files...")

    for folder in ['build', 'dist', f'{PACKAGE_NAME}.egg-info']:
        shutil.rmtree(folder, ignore_errors=True)

    # 清除 __pycache__ 文件夹
    for root, dirs, _ in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                pycache_path = os.path.join(root, d)
                print(f"🗑️ Removing __pycache__: {pycache_path}")
                shutil.rmtree(pycache_path, ignore_errors=True)

def build_package():
    """构建 tar.gz 和 .whl 包"""
    print("🛠️ Building package...")
    run("python -m build")

def upload_to_pypi():
    """上传构建好的包到 PyPI"""
    print("⬆️ Uploading to PyPI...")
    run(f"twine upload --repository {PYPI_REPO} dist/*")

def main():
    clean_previous_builds()
    build_package()
    upload_to_pypi()
    print("✅ 发布完成：已上传到 PyPI")

if __name__ == "__main__":
    main()