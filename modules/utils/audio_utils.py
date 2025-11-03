"""Audio processing utilities."""

import numpy as np


def to_mono_and_normalize(data: np.ndarray) -> np.ndarray:
    """
    將音訊轉單聲道，並正規化為 float32 [-1, 1]。
    
    Args:
        data: 輸入音訊數據（可以是多維陣列）
        
    Returns:
        正規化後的單聲道音訊（float32, [-1, 1]）
    """
    if data.ndim == 1:
        mono = data
    elif data.ndim == 2:
        if data.shape[0] <= 3 and data.shape[1] > 3:
            mono = data.mean(axis=0)
        elif data.shape[1] <= 3 and data.shape[0] > 3:
            mono = data.mean(axis=1)
        else:
            mono = data.mean(axis=-1)
    else:
        mono = data.reshape(-1)

    # 正規化到 float32 [-1, 1]
    if mono.dtype == np.int16:
        mono = mono.astype(np.float32) / 32768.0
    else:
        mono = mono.astype(np.float32, copy=False)
        max_abs = np.max(np.abs(mono)) if mono.size else 0.0
        # 若值明顯超過 1，推測是未縮放的 int16 轉 float32，進行縮放
        if max_abs > 8.0:
            mono = mono / 32768.0
        # 最後裁切避免溢出
        mono = np.clip(mono, -1.0, 1.0)
    return mono
