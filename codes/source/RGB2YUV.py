# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/transforms/functional.py
import numpy as np

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}

def rgb2ycbcr(rgb):
    assert (rgb.dtype == np.float64 or rgb.dtype == np.float32)
    
    r, g, b = np.dsplit(rgb, 3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = np.concatenate((y, cb, cr), axis=-1)
    return ycbcr

def ycbcr2rgb(ycbcr):
    assert (ycbcr.dtype == np.float64 or ycbcr.dtype == np.float32)
    
    y, cb, cr = np.dsplit(ycbcr, 3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=-1)
    return rgb