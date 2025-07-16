# your_package/core.py

from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np
import urllib.request
from tqdm import tqdm
from platformdirs import user_data_dir
from astropy_healpix import HEALPix
from astropy import units as u
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

APP_NAME = "dustmaps3d"
DATA_VERSION = "v2.1"
DATA_FILENAME = f"data_{DATA_VERSION}.parquet"
DATA_URL = f"https://github.com/Grapeknight/dustmaps3d/releases/download/{DATA_VERSION}/{DATA_FILENAME}"
LOCAL_DATA_PATH = Path(user_data_dir(APP_NAME)) / DATA_FILENAME

_HEALPIX = HEALPix(nside=1024, order='ring')

# --- 新增的多进程 Worker 相关部分 ---
# 这部分代码必须在模块的顶层

# 全局变量，用于在每个工作进程中存储加载的数据
# 这样可以避免在每个任务中重复加载数据
worker_df = None

def init_worker():
    """
    Initializer for each worker process.
    This function is called once per worker process when the pool is created.
    It loads the main data file into a global variable `worker_df` for that process.
    """
    global worker_df
    print(f"[dustmaps3d-worker] Initializing worker process {mp.current_process().pid}...")
    # 调用 load_data() 来下载（如果需要）并加载数据
    worker_df = load_data()
    print(f"[dustmaps3d-worker] Worker process {mp.current_process().pid} initialized.")


def dustmaps3d_worker_task(args):
    """
    The actual task function executed by each worker process.
    It takes a chunk of coordinates, processes them in a vectorized way,
    and returns the results.
    """
    l_chunk, b_chunk, d_chunk = args

    # 确保 worker 已经被正确初始化
    if worker_df is None:
        raise RuntimeError("Worker process not initialized. `worker_df` is None.")

    # 这是一个向量化的操作，非常高效
    pix_ids = _HEALPIX.lonlat_to_healpix(l_chunk * u.deg, b_chunk * u.deg)
    rows = worker_df.iloc[pix_ids].copy()
    rows['distance'] = d_chunk
    
    # 直接调用核心的 map 函数来计算结果
    return map(rows)
    

# --- 你的原始代码部分（大部分保持不变） ---

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


@lru_cache(maxsize=1)
def load_data():
    if not LOCAL_DATA_PATH.exists():
        print(f"[dustmaps3d] Downloading {DATA_FILENAME} (~350MB)...")
        LOCAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=DATA_FILENAME) as t:
            urllib.request.urlretrieve(DATA_URL, LOCAL_DATA_PATH, reporthook=t.update_to)
    # 使用 pd.read_parquet 读取数据
    return pd.read_parquet(LOCAL_DATA_PATH)

# ... (bubble_diffuse, component4, diffusion_derived_function, etc. 保持不变) ...
def bubble_diffuse(x, h, b_lim, diffuse_dust_rho, bubble):
    span = 0.01
    span_0 = h / np.sin(np.deg2rad(np.abs(b_lim)))
    Cum_EBV_0 = span_0 * diffuse_dust_rho
    C_0 = Cum_EBV_0 * (1 - np.exp(-bubble / span_0))

    exp_term = np.exp(5 * bubble / span)
    a = 1 / exp_term
    b = 1 / (1 + exp_term)
    c = 0.5
    deta = C_0 / ((1 + a) * (c - b))

    res = np.zeros_like(x)
    mask = x >= bubble
    res[mask] = (Cum_EBV_0 * (1 - np.exp(-x[mask] / span_0)) - C_0)[mask]
    res += deta * (1 + a) * ((1 / (1 + np.exp(-5 * ((x - bubble) / span)))) - b)
    return res


def component4(x, b_lim, bubble, diffuse_dust_rho, h,
               d1, s1, c1, d2, s2, c2, d3, s3, c3, d4, s4, c4):
    def sigmoid_contrib(d, s, c):
        term = d + 2 * s + bubble
        numerator = c * (1 / np.exp(5 * term / s) + 1)
        return numerator, term, s

    terms = [sigmoid_contrib(d, s, c) for d, s, c in [(d1, s1, c1), (d2, s2, c2), (d3, s3, c3), (d4, s4, c4)]]

    result = bubble_diffuse(x, h, b_lim, diffuse_dust_rho, bubble)
    for numerator, term, s in terms:
        result += numerator / (1 + np.exp(-5 * (x - term) / s)) - numerator / (1 + np.exp(5 * term / s))
    return result


def diffusion_derived_function(x, b_lim, diffuse_dust_rho, h):
    span_0 = h / np.sin(np.deg2rad(np.abs(b_lim)))
    return diffuse_dust_rho * np.exp(-x / span_0)


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-b * (x - a)))


def derivative_of_sigmoid(x, a, b, c):
    sig = sigmoid(x, a, b, 1)
    return b * c * sig * (1 - sig)


def sigmoid_of_component(bubble, distance, span, Cum_EBV):
    a = distance + 2 * span + bubble
    b = 5 / span
    c = Cum_EBV * (1 / np.exp(5 * a / span) + 1)
    return a, b, c


def derivative_of_component4(x, b_lim, bubble, diffuse_dust_rho, h,
                             d1, s1, c1, d2, s2, c2, d3, s3, c3, d4, s4, c4):
    a1, b1, c1 = sigmoid_of_component(bubble, d1, s1, c1)
    a2, b2, c2 = sigmoid_of_component(bubble, d2, s2, c2)
    a3, b3, c3 = sigmoid_of_component(bubble, d3, s3, c3)
    a4, b4, c4 = sigmoid_of_component(bubble, d4, s4, c4)

    base = np.zeros_like(x)
    mask = x >= bubble
    base[mask] = diffusion_derived_function(x[mask], b_lim[mask], diffuse_dust_rho[mask], h[mask])

    return base + \
        derivative_of_sigmoid(x, a1, b1, c1) + \
        derivative_of_sigmoid(x, a2, b2, c2) + \
        derivative_of_sigmoid(x, a3, b3, c3) + \
        derivative_of_sigmoid(x, a4, b4, c4)


def map(df):
    distance = df['distance'].fillna(df['max_distance'])
    EBV = component4(distance, df['b_lim'], df['bubble'], df['diffuse_dust_rho'], df['h'],
                     df['distance_1'], df['span_1'], df['Cum_EBV_1'],
                     df['distance_2'], df['span_2'], df['Cum_EBV_2'],
                     df['distance_3'], df['span_3'], df['Cum_EBV_3'],
                     df['distance_4'], df['span_4'], df['Cum_EBV_4'])

    dust = derivative_of_component4(distance, df['b_lim'], df['bubble'], df['diffuse_dust_rho'], df['h'],
                                    df['distance_1'], df['span_1'], df['Cum_EBV_1'],
                                    df['distance_2'], df['span_2'], df['Cum_EBV_2'],
                                    df['distance_3'], df['span_3'], df['Cum_EBV_3'],
                                    df['distance_4'], df['span_4'], df['Cum_EBV_4'])

    sigma_finally = np.empty_like(df['sigma'], dtype=float)

    mask1 = distance < 1
    sigma_finally[mask1] = np.nanmin([df['sigma'][mask1], df['sigma_0_2'][mask1]], axis=0)

    mask2 = (distance >= 1) & (distance < 2)
    sigma_finally[mask2] = np.nanmin([df['sigma'][mask2], df['sigma_0_2'][mask2], df['sigma_1_4'][mask2]], axis=0)

    mask3 = (distance >= 2) & (distance < 4)
    sigma_finally[mask3] = np.nanmin([df['sigma_1_4'][mask3], df['sigma_2_max'][mask3]], axis=0)

    sigma_finally[distance >= 4] = df['sigma_2_max'][distance >= 4]

    # map 函数现在返回一个包含4个numpy数组的元组
    return EBV.values, dust.values, sigma_finally, df['max_distance'].values


# --- 删除旧的、低效的 _dustmaps3d_worker ---
# def _dustmaps3d_worker(args):
#     ...


def dustmaps3d(l, b, d, n_process: int = None):
    """
    3D dust map (Wang et al. 2025).

    Parameters
    ----------
    l : np.ndarray
        Galactic longitude in degrees.
    b : np.ndarray
        Galactic latitude in degrees.
    d : np.ndarray
        Distance in kpc.
    n_process : int, optional
        Number of parallel processes to use. If None (default), the function runs in single-threaded mode.
        When set to an integer >= 1, the input data is split evenly across processes, and
        each process independently computes the dust values in parallel.

    Returns
    -------
    EBV : np.ndarray
        E(B–V) extinction value along the line of sight.
    dust : np.ndarray
        Dust density (d(EBV)/dx) in mag/kpc.
    sigma : np.ndarray
        Estimated uncertainty in E(B–V).
    max_distance : np.ndarray
        Maximum reliable distance along the line of sight for each direction.

    Notes
    -----
    - When using `n_process`, make sure `l`, `b`, `d` are arrays of equal length.
    - This function uses `multiprocessing.Pool` internally and is safe for CPU-bound batch queries.
    """
    l = np.atleast_1d(l)
    b = np.atleast_1d(b)
    d = np.atleast_1d(d)

    if not (len(l) == len(b) == len(d)):
        raise ValueError("l, b, d must be the same length")

    if np.isnan(l).any() or np.isnan(b).any():
        print("[dustmaps3d] Error: Input `l` and `b` must not contain NaN values.")
        raise ValueError("NaN values detected in `l` or `b`. These are not supported by HEALPix mapping.")

    # --- 单进程模式 (修改以匹配 map 的新返回值) ---
    if n_process is None or n_process < 2 or len(l) <= 1:
        df = load_data()
        pix_ids = _HEALPIX.lonlat_to_healpix(l * u.deg, b * u.deg)
        rows = df.iloc[pix_ids].copy()
        rows['distance'] = d
        # map现在返回numpy数组，这很好
        return map(rows)

    # --- 多进程模式 (完全重写) ---
    else:
        # 使用 'spawn' 上下文来确保跨平台的一致性和安全性
        # 这是解决 Windows 问题的关键
        ctx = mp.get_context("spawn")
        
        # 将输入数据按进程数分块
        chunks = np.array_split(np.arange(len(l)), n_process)
        # 为每个 worker 准备参数 (每个 worker 接收一个数据块)
        args_list = [(l[chunk], b[chunk], d[chunk]) for chunk in chunks if len(chunk) > 0]

        # 创建一个 Pool，并指定 initializer
        # initializer=init_worker 会在每个工作进程启动时调用 init_worker()
        with ctx.Pool(processes=n_process, initializer=init_worker) as pool:
            # pool.map 会将 args_list 中的每个元素传递给 dustmaps3d_worker_task
            # results 将是一个列表，每个元素是 map() 的返回值 (一个元组)
            results = pool.map(dustmaps3d_worker_task, args_list)

        # 合并来自所有 worker 的结果
        if not results:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # `results` 是一个列表，例如：[(ebv1, dust1, ...), (ebv2, dust2, ...)]
        # 使用 zip(*) 将其转置为 [(ebv1, ebv2), (dust1, dust2), ...]
        ebv_list, dust_list, sigma_list, maxd_list = zip(*results)

        # 使用 np.concatenate 将每个部分连接成一个大的 numpy 数组
        return (
            np.concatenate(ebv_list),
            np.concatenate(dust_list),
            np.concatenate(sigma_list),
            np.concatenate(maxd_list)
        )