import numpy as np
from PIL import Image


def X2Cube(img):
    img = np.asarray(img)
    B = skip = [4, 4]
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    img = out.reshape(M // 4, N // 4, 16)
    img = img / img.max() * 255
    img = img.astype(np.uint8)
    return img


def read_hsi(image_path: str):
    image = Image.open(image_path)
    image = X2Cube(image)
    return image
