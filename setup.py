from setuptools import setup, find_packages

setup(
    name="3d_dust_maps",
    version="1.0.0",
    description="An all-sky 3D dust map Based on Gaia and LAMOST.",
    author="Wang Tao",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "astropy",
        "astropy-healpix"
    ],
    python_requires=">=3.8",
)
