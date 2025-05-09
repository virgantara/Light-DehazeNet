import cv2
import numpy as np
import scipy.ndimage
import time


def cal_depth_map(image, r=15, sigma=0.041337):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    noise = np.random.normal(0, sigma, s.shape)
    raw_depth = 0.121779 + 0.959710 * v - 0.780245 * s + noise
    refined_depth = scipy.ndimage.minimum_filter(raw_depth, (r, r))
    return refined_depth


def guided_filter(I, p, r, eps):
    I = I.astype(np.float32) / 255.0
    p = p.astype(np.float32)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    corr_I = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    q = mean_a * I + mean_b
    return q


def estimate_airlight(img, depth_map, top_percent=0.001):
    h, w = depth_map.shape
    flat = depth_map.flatten()
    img_flat = img.reshape(-1, 3)

    n_select = int(h * w * top_percent)
    indices = flat.argsort()[-n_select:]
    brightest = img_flat[indices]
    return np.max(brightest, axis=0)


def recover_scene_radiance(I, t, A, t0=0.05):
    I = I.astype(np.float32) / 255.0
    t = np.clip(t, t0, 1.0)[:, :, np.newaxis]
    J = (I - A) / t + A
    J = np.clip(J * 255.0, 0, 255).astype(np.uint8)
    return J


def cap_dehaze(image_bgr, beta=1.0, r=15, guided_r=60, eps=1e-3):
    start = time.time()

    depth_map = cal_depth_map(image_bgr, r)
    refine_depth = guided_filter(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), depth_map, guided_r, eps)
    transmission = np.exp(-beta * refine_depth)

    A = estimate_airlight(image_bgr, depth_map)
    dehazed = recover_scene_radiance(image_bgr, transmission, A / 255.0)

    print(f"[CAP] Execution time: {time.time() - start:.3f} sec")
    return dehazed
