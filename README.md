**Read this in: [English](README.md) | [中文](README.zh-CN.md)**


# dustmaps3d

**Note**: This is a fork of the original [Grapeknight/dustmaps3d](https://github.com/Grapeknight/dustmaps3d) project. This version introduces performance improvements through multiprocessing and adds a convenient command-line interface (CLI) for batch processing.

To run directly from GitHub using `uvx`, please ensure you have `uv` installed. You can find the installation guide in the [official `uv` documentation](https://github.com/astral-sh/uv).

🌌 **An all-sky 3D dust extinction map based on Gaia and LAMOST**

📄 *Wang et al. (2025),* *An all-sky 3D dust map based on Gaia and LAMOST*  
📌 DOI: [10.12149/101620](https://doi.org/10.12149/101620)

📦 *A Python package for easy access to the 3D dust map*  
📌 DOI: [10.12149/101619](https://nadc.china-vo.org/res/r101619/)

---

## 📦 Installation

Install via pip:

```bash
pip install dustmaps3d
```

**Note:** Installing the package does *not* include the data file.  
The ~350 MB model data will be **automatically downloaded** from GitHub on **first use**.  
⚠️ If you experience network issues when downloading from GitHub,  
you can manually download the data from NADC:  
🔗 https://nadc.china-vo.org/res/r101619/

---

## 🚀 Usage

```python
from dustmaps3d import dustmaps3d

l = [120.0]
b = [30.0]
d = [1.5]

EBV, dust, sigma, max_d = dustmaps3d(l, b, d)
print(f"EBV: {EBV.values[0]:.4f} [mag]")
print(f"Dust: {dust.values[0]:.4f} [mag/kpc]")
print(f"Sigma: {sigma.values[0]:.4f} [mag]")
print(f"Max distance: {max_d.values[0]:.4f} kpc")

```

**Batch example with FITS:**

```python
import numpy as np
from astropy.table import Table
from dustmaps3d import dustmaps3d

data = Table.read('input.fits')   
l = data['l'].astype(float)
b = data['b'].astype(float)
d = data['d'].astype(float)

EBV, dust, sigma, max_d = dustmaps3d(l, b, d)

data['EBV_3d'] = EBV
data['dust'] = dust
data['sigma'] = sigma
data['max_distance'] = max_d
data.write('output.fits', overwrite=True)
```

**Batch Processing with Pandas DataFrame**

For integration into Python workflows, the `dustmaps3d_from_df` function has been added. It leverages multiprocessing to efficiently handle large DataFrames.

```python
import pandas as pd
from dustmaps3d import dustmaps3d_from_df

# Example with a large DataFrame (30 million rows)
data = {
    'l': [120.0, 80.5, 210.1] * 10000000,
    'b': [30.0, -15.2, 45.5] * 10000000,
    'd': [1.5, 0.8, 3.0] * 10000000
}
df = pd.DataFrame(data)

# Process the DataFrame using 16 cores.
# You can customize the number of rows each core handles via 'chunk_size'.
processed_df = dustmaps3d_from_df(df, n_process=16, chunk_size=100000)

# Save the results to a new CSV file
processed_df.to_csv('processed_dustmaps3d.csv', index=False)
```

**Command-Line Interface (CLI)**

You can now process a CSV file directly from your terminal.

First, you can install it system-wide using pipx, install it locally (pip install . in the project root), or use uvx for a direct, installation-free execution.

```bash
# Install from GitHub using pipx
pipx install git+https://github.com/SunnyHina/dustmaps3d.git

# Usage: dust <input_file> <output_file> [--threads <number_of_threads>]

# Process a file using 8 threads
dust input.csv output.csv --threads 8

# Or run directly from GitHub without installation using uvx
uvx --from git+https://github.com/SunnyHina/dustmaps3d.git dust input.csv output.csv --threads 8
```

Your `input.csv` must contain the columns: `l` (Galactic longitude), `b` (Galactic latitude), and `d` (distance in kpc).

---


## 🧠 Function Description

### `dustmaps3d(l, b, d, n_process=None)`

Estimates 3D dust extinction and related quantities for given galactic coordinates and distances.

| Input       | Type         | Description                        | Unit     |
|-------------|--------------|------------------------------------|----------|
| `l`         | np.ndarray   | Galactic longitude                 | degrees  |
| `b`         | np.ndarray   | Galactic latitude                  | degrees  |
| `d`         | np.ndarray   | Distance                           | kpc      |

#### Returns:

| Output       | Type         | Description                           | Unit     |
|--------------|--------------|---------------------------------------|----------|
| `EBV`        | np.ndarray   | E(B–V) extinction                     | mag      |
| `dust`       | np.ndarray   | Dust density (d(EBV)/dx)             | mag/kpc  |
| `sigma`      | np.ndarray   | Estimated uncertainty in E(B–V)      | mag      |
| `max_d`      | np.ndarray   | Maximum reliable distance            | kpc      |

> If `d` contains `NaN`, it will be automatically replaced by the maximum reliable distance along that line of sight (`max_d`).
> 
> If the input `d` exceeds `max_d`, it indicates the point lies beyond the model's reliable range. The returned values in this case are extrapolated and **not guaranteed to be accurate**.

---

## ⚡ Performance

- Fully vectorized and optimized with NumPy
- On a modern personal computer, evaluating **100 million stars takes only ~100 seconds**

---

## 📂 Data Version

This version uses `data_v2.2.parquet`, released under [v2.2](https://github.com/Grapeknight/dustmaps3d/releases/tag/v2.2)

---

## 📜 Citation

If you use this model or the Python package, please cite both:

- Wang, T. (2025), *An all-sky 3D dust map based on Gaia and LAMOST*. DOI: [10.12149/101620](https://doi.org/10.12149/101620)  
- *dustmaps3d: A Python package for easy access to the 3D dust map*. DOI: [10.12149/101619](https://nadc.china-vo.org/res/r101619/)

---

## 🛠️ License

This project is open-source and distributed under the MIT License.

---

## 📫 Contact

If you have any questions, suggestions, or encounter issues using this package,  
please feel free to contact the authors via GitHub issues or email.

- Prof. Yuan Haibo: yuanhb@bnu.edu.cn  
- Wang Tao: wt@mail.bnu.edu.cn

🔗 [GitHub Repository](https://github.com/Grapeknight/dustmaps3d)