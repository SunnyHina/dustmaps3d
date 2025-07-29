**Read this in: [English](README.md) | [中文](README.zh-CN.md)**

# dustmaps3d

**注意**：本项目是 [Grapeknight/dustmaps3d](https://github.com/Grapeknight/dustmaps3d) 的一个 fork。此版本通过引入多进程计算提升性能，并增加了一个方便的命令行工具（CLI）用于批量处理。

如果需要使用 uvx 直接从 GitHub 运行，请确保您已经安装了 `uv`。您可以访问 [uv 官方文档](https://github.com/astral-sh/uv) 获取安装指南。

🌌 **基于 Gaia 和 LAMOST 构建的全天三维尘埃消光图**

📄 *Wang et al. (2025)，An all-sky 3D dust map based on Gaia and LAMOST*  
📌 DOI: [10.12149/101620](https://doi.org/10.12149/101620)

📦 *A Python package for easy access to the 3D dust map*   
📌 DOI: [10.12149/101619](https://nadc.china-vo.org/res/r101619/)

---

## 📦 安装

通过 pip 安装：

```bash
pip install dustmaps3d
```

**注意：** 安装包本身并不包含模型数据。  
约 350MB 的数据文件将在**首次使用时自动从 GitHub 下载**。 

国内下载可能需要科学上网 QAQ

⚠️ 若遇到网络连接问题，也可从 NADC 手动下载数据：  
🔗 https://nadc.china-vo.org/res/r101619/

---

## 🚀 使用示例

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

**FITS 文件批量处理示例：**

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

**使用 Pandas DataFrame 进行批量处理**

为了方便地集成到 Python 工作流中，我们添加了 `dustmaps3d_from_df` 函数。它利用多进程来高效地处理大规模的 Pandas DataFrame。

```python
import pandas as pd
from dustmaps3d import dustmaps3d_from_df

# 一个处理大规模 DataFrame (三千万行) 的示例
data = {
    'l': [120.0, 80.5, 210.1] * 10000000,
    'b': [30.0, -15.2, 45.5] * 10000000,
    'd': [1.5, 0.8, 3.0] * 10000000
}
df = pd.DataFrame(data)

# 使用 16 个核心处理 DataFrame。
# 你可以通过 'chunk_size' 参数自定义每个核心处理的数据条数。
processed_df = dustmaps3d_from_df(df, n_process=16, chunk_size=100000)

# 将处理结果保存到新的 CSV 文件
processed_df.to_csv('processed_dustmaps3d.csv', index=False)
```

**通过命令行使用**

现在你可以直接在终端中处理 CSV 文件。

首先，请确保已经安装了本工具包（在项目根目录运行 `pip install .`），或者使用 `uvx` 来免安装直接运行。

```bash
# 用法: dust <输入文件> <输出文件> [--threads <线程数>]

# 使用 8 个线程处理文件
dust input.csv output.csv --threads 8

# 或使用 uvx 直接从 GitHub 运行，无需安装
uvx --from git+https://github.com/SunnyHina/dustmaps3d.git dust input.csv output.csv --threads 8
```

您的 `input.csv` 文件必须包含以下列：`l` (银经), `b` (银纬), 和 `d` (距离, 单位 kpc)。

---
## 🧠 函数说明

### `dustmaps3d(l, b, d)`

根据输入的银河坐标 `(l, b)` 和距离 `d`，返回对应的尘埃消光信息。

| 输入         | 类型         | 描述                        | 单位     |
|--------------|--------------|-----------------------------|----------|
| `l`          | np.ndarray   | 银经                      | 度       |
| `b`          | np.ndarray   | 银纬                      | 度       |
| `d`          | np.ndarray   | 距离                      | kpc      |

#### 返回：

| 输出         | 类型         | 描述                              | 单位     |
|--------------|--------------|-----------------------------------|----------|
| `EBV`        | np.ndarray   | E(B–V) 消光值                     | mag      |
| `dust`       | np.ndarray   | 尘埃密度（d(EBV)/dx）             | mag/kpc  |
| `sigma`      | np.ndarray   | E(B–V) 的不确定度估计             | mag      |
| `max_d`      | np.ndarray   | 每条视线方向上可靠的最大距离      | kpc      |

> 如果输入的 `d` 中包含 `NaN`，程序将自动将其替换为该视线方向的最大可靠距离（`max_d`）。
>
> 如果输入的 `d` 超过了 `max_d`，则说明该点超出了模型的可靠范围。此时返回的值是通过外推计算的，**不具有可靠性**。

---

## ⚡ 性能

- 基于 NumPy 完全向量化实现
- 在普通个人计算机上，单线程处理 **一亿颗恒星** 仅需约 **100 秒**

---

## 📂 数据版本

当前版本使用数据文件：`data_v2.2.parquet`，来自发布版本 [v2.2](https://github.com/Grapeknight/dustmaps3d/releases/tag/v2.2)

---

## 📜 引用说明

如果您在研究中使用了该模型或包，请引用以下两项：

- Wang, T. (2025), *An all-sky 3D dust map based on Gaia and LAMOST*  
  DOI: [10.12149/101620](https://doi.org/10.12149/101620)
- *dustmaps3d: A Python package for easy access to the 3D dust map*  
  DOI: [10.12149/101619](https://nadc.china-vo.org/res/r101619/)

---

## 🛠️ 授权协议

MIT License

## 📫 联系方式

如在使用本工具过程中有任何问题、建议或技术交流，欢迎通过 GitHub issue 或邮箱联系作者团队：

- Prof. Yuan Haibo（苑海波 教授）: yuanhb@bnu.edu.cn  
- Wang Tao（王涛）: wt@mail.bnu.edu.cn  

🔗 [GitHub Repository](https://github.com/Grapeknight/dustmaps3d)