# dustmaps3d/core.py
import os
import pandas as pd
import numpy as np
import urllib.request
from tqdm import tqdm
from astropy_healpix import HEALPix
from astropy import units as u

# 远程数据地址
DATA_URL = "https://github.com/Grapeknight/dustmaps3d/releases/download/v1.0/data_v1.parquet"
LOCAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "data_v1.parquet")

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_data_if_needed():
    if not os.path.exists(LOCAL_DATA_PATH):
        print("Downloading 3D dust map data... (~686MB)")
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="data_v1.parquet") as t:
            urllib.request.urlretrieve(DATA_URL, LOCAL_DATA_PATH, reporthook=t.update_to)
    return pd.read_parquet(LOCAL_DATA_PATH)

df = download_data_if_needed()

# 以下保留原始函数定义

def bubble_diffuse(x,h,b_lim,diffuse_dust_rho,bubble): 
    span = 0.01
    span_0 = h / np.sin(np.deg2rad(np.abs(b_lim)))
    Cum_EBV_0 = span_0 * diffuse_dust_rho
    C_0 = Cum_EBV_0 * (1 - np.exp(- (bubble) / span_0))
    f = (Cum_EBV_0 * (1 - np.exp(-x / span_0))) - C_0
    a = 1/np.exp(5 * bubble /span)
    b = 1 / (1 + np.exp(5 * bubble/span))
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
    mask = distance < 1
    sigma_finally[mask] = np.nanmin(np.array([df['sigma'][mask], df['sigma_0_2'][mask]]), axis=0)
    mask = (distance >= 1) & (distance < 2)
    sigma_finally[mask] = np.nanmin(np.array([df['sigma'][mask], df['sigma_0_2'][mask], df['sigma_1_4'][mask]]), axis=0)
    mask = (distance >= 2) & (distance < 4)
    sigma_finally[mask] = np.nanmin(np.array([df['sigma_1_4'][mask], df['sigma_2_max'][mask]]), axis=0)
    mask = distance >= 4
    sigma_finally[mask] = df['sigma_2_max'][mask]
    return EBV, dust, sigma_finally, df['max_distance']

def dustmaps3d(l,b,d):
    l = np.atleast_1d(l)
    b = np.atleast_1d(b)
    d = np.atleast_1d(d)
    healpix = HEALPix(nside=1024, order='ring')
    pix_ids = healpix.lonlat_to_healpix(l * u.deg, b * u.deg)
    rows = df.iloc[pix_ids].copy()
    rows['distance'] = d
    EBV, dust, sigma_finally, max_d = map(rows)
    return EBV, dust, sigma_finally, max_d
