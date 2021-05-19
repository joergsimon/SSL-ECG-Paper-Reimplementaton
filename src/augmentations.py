import numpy as np
import numpy.random as rd
import cv2
from enum import Enum


class AugmentationTypes(Enum):
    ORIGINAL = 0
    ADD_NOISE = 1
    SCALE = 2
    NEGATE = 3
    TEMP_INV = 4
    PERMUTATE = 5
    TIME_WRAP = 6


def add_noise(signal, SNR, *args, **kwargs):
    watts = signal**2
    sig_avg_watts = np.mean(watts)
    sig_avg_db = 10*np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - SNR
    noise_avg_watts = 10** (noise_avg_db / 10)
    noise = rd.normal(0.0, np.sqrt(noise_avg_watts), len(signal))
    noisy_signal = signal + noise
    return noisy_signal


def scale(signal, beta, *args, **kwargs):
    scaled = signal * beta
    return scaled


def negate(signal, *args, **kwargs):
    negated = -1 * signal
    return negated


def temp_invert(signal, *args, **kwargs):
    t_inv = np.flip(signal).copy()
    return t_inv


def _get_sections(signal, n_sections, *args, **kwargs):
    s_len = len(signal) // n_sections
    sections = [signal[i * s_len:i * s_len + s_len] for i in range(n_sections - 1)]
    sections.append(signal[(n_sections - 1) * s_len:])
    return sections


def permuatate(signal, n_sections, *args, **kwargs):
    sections = _get_sections(signal, n_sections, args, kwargs)
    rd.shuffle(sections)
    sec_arr = np.concatenate(sections, axis=0)
    permutation = sec_arr.reshape(signal.shape)
    return permutation


def time_warp(signal, n_sections, k, *args, **kwargs):
    sections = _get_sections(signal, n_sections, args, kwargs)
    idxes = list(range(n_sections))
    strech = rd.choice(idxes, n_sections//2, replace=False)
    ks = [k if i in strech else 1/k for i in range(n_sections)]
    warped_sections = []
    for factor, s in zip(ks, sections):
        l = int(np.ceil(len(s)*factor))
        warped = cv2.resize(s, (1,l), interpolation=cv2.INTER_LINEAR).reshape((l,))
        warped_sections += list(warped)

    if len(warped_sections) < len(signal):
        warped_sections += [0] * (len(signal) - len(warped_sections))

    res = np.array(warped_sections)
    if len(res) > len(signal):
        res = res[:len(signal)]
    return res
