from setuptools import setup, find_packages

setup(
    name="dustmaps3d",
    version="2.0.0",  # 重要：更新版本号
    description="An all-sky 3D dust map based on Gaia and LAMOST.",
    author="Wang Tao",
    author_email="1026579743@qq.com",
    url="https://github.com/Grapeknight/dustmaps3d",
    packages=find_packages(),
    include_package_data=False,
    install_requires=[
        "numpy",
        "pandas",
        "astropy",
        "astropy-healpix"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
