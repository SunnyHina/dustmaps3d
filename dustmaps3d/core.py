from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np
import urllib.request
from tqdm import tqdm
from platformdirs import user_data_dir
from astropy_healpix import HEALPix
from astropy import units as u
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

APP_NAME = "dustmaps3d"
DATA_VERSION = "v2.1"
DATA_FILENAME = f"data_{DATA_VERSION}.parquet"
DATA_URL = f"https://github.com/Grapeknight/dustmaps3d/releases/download/{DATA_VERSION}/{DATA_FILENAME}"
LOCAL_DATA_PATH = Path(user_data_dir(APP_NAME)) / DATA_FILENAME

_HEALPIX = HEALPix(nside=1024, order='ring')


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
    return pd.read_parquet(LOCAL_DATA_PATH)


def bubble_diffuse(x,h,b_lim,diffuse_dust_rho,bubble): 
    span = 0.01
    span_0 = h / np.sin(np.deg2rad(np.abs(b_lim)))
    Cum_EBV_0 = span_0 * diffuse_dust_rho
    C_0 = Cum_EBV_0 * (1 - np.exp(- (bubble) / span_0))
    f = (Cum_EBV_0 * (1 - np.exp(-x / span_0))) - C_0
    exp_n = np.exp(5 * bubble /span)
    a = 1 / exp_n
    b = 1 / (1 + exp_n)
    c = 0.5
    deta = C_0/((1+a)*(c-b))
    return np.where(x < (bubble), 0, f) + deta*(1+a)*((1 / (1 + np.exp(-5 * ((x - bubble)/span))) )-b)

def component4(x, b_lim, bubble, diffuse_dust_rho, h, distance_1, span_1, Cum_EBV_1, distance_2, span_2, Cum_EBV_2, distance_3, span_3, Cum_EBV_3, distance_4, span_4, Cum_EBV_4):
    Numerator_1 = Cum_EBV_1*(1/np.exp(5 * (distance_1 + (span_1*2) + bubble) /span_1) + 1)
    Numerator_2 = Cum_EBV_2*(1/np.exp(5 * (distance_2 + (span_2*2) + bubble)/span_2) + 1)
    Numerator_3 = Cum_EBV_3*(1/np.exp(5 * (distance_3 + (span_3*2) + bubble)/span_3) + 1)
    Numerator_4 = Cum_EBV_4*(1/np.exp(5 * (distance_4 + (span_4*2) + bubble)/span_4) + 1)
    
    return (bubble_diffuse(x,h,b_lim,diffuse_dust_rho,bubble)
                     
                    +((Numerator_1/ (1 + np.exp(-5 * ((x) - (distance_1 + (span_1*2) + bubble))/span_1))) 
                    -(Numerator_1 / (1 + np.exp(5 * (distance_1 + (span_1*2) + bubble)/span_1))))
                    
                    +((Numerator_2 / (1 + np.exp(-5 * ((x) - (distance_2 + (span_2*2) + bubble))/span_2))) 
                    -(Numerator_2 / (1 + np.exp(5 * ((distance_2 + (span_2*2) + bubble))/span_2))))

                    +((Numerator_3 / (1 + np.exp(-5 * ((x) - (distance_3 + (span_3*2) + bubble))/span_3))) 
                    -(Numerator_3 / (1 + np.exp(5 * ((distance_3 + (span_3*2) + bubble))/span_3))))

                    +((Numerator_4 / (1 + np.exp(-5 * ((x) - (distance_4 + (span_4*2) + bubble))/span_4))) 
                    -(Numerator_4 / (1 + np.exp(5 * ((distance_4 + (span_4*2) + bubble))/span_4))))
                    )       
 
def diffusion_derived_function(x, b_lim, diffuse_dust_rho, h ):
    span_0 = h / np.sin(np.deg2rad(np.abs(b_lim)))
    return diffuse_dust_rho * (np.exp(- x / span_0))

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-b * (x - a)))

def derivative_of_sigmoid(x, a, b, c):
    return b * c * sigmoid(x, a, b, 1) * (1 - (sigmoid(x, a, b, 1)))

def sigmoid_of_component(bubble, distance, span, Cum_EBV):
    a = distance + (2*span) + bubble
    b = 5 / span
    c = Cum_EBV*(1/np.exp(5 * a /span) + 1)
    return a, b, c

def derivative_of_component4(x, b_lim, bubble, diffuse_dust_rho, h, distance_1, span_1, Cum_EBV_1, distance_2, span_2, Cum_EBV_2, distance_3, span_3, Cum_EBV_3, distance_4, span_4, Cum_EBV_4):
    a_1, b_1, c_1 = sigmoid_of_component(bubble, distance_1, span_1, Cum_EBV_1)
    a_2, b_2, c_2 = sigmoid_of_component(bubble, distance_2, span_2, Cum_EBV_2)
    a_3, b_3, c_3 = sigmoid_of_component(bubble, distance_3, span_3, Cum_EBV_3)
    a_4, b_4, c_4 = sigmoid_of_component(bubble, distance_4, span_4, Cum_EBV_4)
    return (np.where(x < bubble, 0, diffusion_derived_function(x, b_lim, diffuse_dust_rho, h)) 
            + derivative_of_sigmoid(x, a_1, b_1, c_1) 
            + derivative_of_sigmoid(x, a_2, b_2, c_2) 
            + derivative_of_sigmoid(x, a_3, b_3, c_3) 
            + derivative_of_sigmoid(x, a_4, b_4, c_4) 
            )

def read_map(df):
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
    mask = distance < 1
    sigma_finally[mask] = np.nanmin(np.array([df['sigma'][mask], df['sigma_0_2'][mask]]), axis=0)
    mask = (distance >= 1) & (distance < 2)
    sigma_finally[mask] = np.nanmin(np.array([df['sigma'][mask], df['sigma_0_2'][mask], df['sigma_1_4'][mask]]), axis=0)
    mask = (distance >= 2) & (distance < 4)
    sigma_finally[mask] = np.nanmin(np.array([df['sigma_1_4'][mask], df['sigma_2_max'][mask]]), axis=0)
    mask = distance >= 4
    sigma_finally[mask] = df['sigma_2_max'][mask]
    return EBV, dust, sigma_finally, df['max_distance']



def _dustmaps3d_worker(args):
    l_chunk, b_chunk, d_chunk = args
    results = [dustmaps3d(l, b, d) for l, b, d in zip(l_chunk, b_chunk, d_chunk)]
    return list(zip(*results))  # (EBV, dust, sigma, max_d)

def dustmaps3d(l, b, d):
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
    
    df = load_data()
    pix_ids = _HEALPIX.lonlat_to_healpix(l * u.deg, b * u.deg)
    rows = df.iloc[pix_ids].copy()
    rows['distance'] = d
    EBV, dust, sigma_finally, max_d = read_map(rows)
    return EBV, dust, sigma_finally, max_d


def plot_dust_section(
    axis1, range1,
    axis2, range2,
    fixed_axis, fixed_value,
    resolution=0.02,
    smooth_sigma=0.5,
    vmin=0.01,
    vmax=5,
    norm_type='log',  # 'log' or 'linear'
    colorbar_ticks=None,
    cmap='Spectral_r',
    figsize=(8, 6),
    save=False,
    show=True,
    filename=None
):
    """
    绘制银河尘埃分布图的任意二维平面切片（基于 dustmaps3d）。

    参数：
    - axis1, axis2: 要绘制的两个轴（'x', 'y', 'z'）
    - range1, range2: 对应轴的范围，例如 [-4, 4]
    - fixed_axis: 第三个固定轴（'x', 'y', 'z'）
    - fixed_value: 该轴的切片位置 [kpc]
    - resolution: 网格分辨率 [kpc]
    - smooth_sigma: 高斯平滑标准差 [单位：像素]
    - vmin, vmax: 色条的最小最大值
    - norm_type: 色条归一化类型：'log' 或 'linear'
    - colorbar_ticks: 色条刻度（可选 list）
    - cmap: 色图（如 'Spectral_r'）
    - figsize: 图像大小（单位英寸）
    - save: 是否保存图像
    - filename: 保存路径（如 'yz_slice.png'）
    """
    import matplotlib.pyplot as plt
    from astropy.coordinates import SkyCoord, CartesianRepresentation
    from matplotlib.colors import LogNorm, Normalize
    from scipy.ndimage import gaussian_filter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # 1. 检查维度合法性
    all_axes = {'x', 'y', 'z'}
    assert {axis1, axis2, fixed_axis} == all_axes, "axis 输入必须覆盖 x, y, z 各一维"

    # 2. 构造二维网格
    vals1 = np.arange(range1[0] + resolution/2, range1[1], resolution)
    vals2 = np.arange(range2[0] + resolution/2, range2[1], resolution)
    A, B = np.meshgrid(vals1, vals2, indexing='ij')
    coords = {'x': None, 'y': None, 'z': None}
    coords[axis1] = A
    coords[axis2] = B
    coords[fixed_axis] = np.full_like(A, fixed_value)

    # 3. 构建 DataFrame
    grid = np.vstack([coords['x'].ravel(), coords['y'].ravel(), coords['z'].ravel()]).T
    df = pd.DataFrame(grid, columns=['x', 'y', 'z'])

    # 4. 转换为 SkyCoord
    cart = CartesianRepresentation(df['x'].values * u.kpc,
                                   df['y'].values * u.kpc,
                                   df['z'].values * u.kpc)
    skycoord = SkyCoord(cart, frame='galactic')
    df['l'] = skycoord.l.deg
    df['b'] = skycoord.b.deg
    df['d'] = skycoord.distance.kpc

    # 5. 查询 dustmaps3d
    l, b, d = df['l'].values, df['b'].values, df['d'].values
    _, dust, _, max_d = dustmaps3d(l, b, d)
    df['dust'] = np.asarray(dust)
    mask = df['d'].to_numpy() <= np.asarray(max_d)
    df = df.loc[mask]

    # 6. 转换为二维图像
    pivot = df.pivot_table(index=axis2, columns=axis1, values='dust')
    smoothed = gaussian_filter(pivot.values, sigma=smooth_sigma)

    if show:
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # 颜色映射控制
        if norm_type == 'log':
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif norm_type == 'linear':
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            raise ValueError("norm_type 只能为 'log' 或 'linear'")

        # 图像绘制
        img = ax.imshow(smoothed,
                        extent=[pivot.columns.min(), pivot.columns.max(),
                                pivot.index.min(), pivot.index.max()],
                        origin='lower',
                        aspect='equal',
                        cmap=cmap,
                        norm=norm)

        # colorbar 设置
        cbar = fig.colorbar(img, cax=cax)
        cbar.set_label('Dust [mag/kpc]')
        if colorbar_ticks is not None:
            cbar.set_ticks(colorbar_ticks)
            cbar.set_ticklabels([f"{v:.2g}" for v in colorbar_ticks])

        # 轴标签与标题
        ax.set_xlabel(f"{axis1.upper()} [kpc]")
        ax.set_ylabel(f"{axis2.upper()} [kpc]")
        ax.set_title(f"Dust in {axis1.upper()}{axis2.upper()} Plane at {fixed_axis} = {fixed_value} kpc")

        # 保存
        if save and filename:
            plt.savefig(filename, dpi=300)
            print(f"图像已保存为 {filename}")

        plt.tight_layout()
        plt.show()
    return smoothed
