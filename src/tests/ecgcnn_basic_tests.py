import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import EcgCNN, EcgHead, EcgNetwork
import src.augmentations as aug
import src.utils as u


def test_zeros(n=0, add_batch_dim=True):
    zero_input = np.zeros((1, n))
    zero_input = torch.from_numpy(zero_input).float()
    if add_batch_dim:
        zero_input = zero_input.unsqueeze(dim=0)
    return zero_input


def test_cnn_basic_dimensions():
    zero_input = test_zeros(n=2560)
    model = EcgCNN()
    model.debug_dim = True
    out = model(zero_input)

    assert out.shape[1] == 128


def test_single_head_loss():
    zero_input = test_zeros(n=128, add_batch_dim=False)
    head = EcgHead()
    out = head(zero_input)
    loss = nn.BCELoss()
    y = u.y_to_torch([1.], shape=(1,1))
    print(out.shape, y.shape)
    l = loss(out, y)
    l.backward()


def test_heads_loss():
    zero_input = test_zeros(n=128, add_batch_dim=False)
    ys = [1,0,0]
    coefficient = [1.0, 1.0, 1.0]
    fake = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU()
    )
    out = fake(zero_input)
    heads = [EcgHead() for _ in range(3)]
    out_list = [h(out) for h in heads]
    loss = nn.BCELoss()
    total_loss = sum([loss(o, u.y_to_torch([y], shape=(1,1))) * c for o, y, c in zip(out_list, ys, coefficient)])
    total_loss.backward()
    print('cool')


def test_ecg_network():
    zero_input = test_zeros(n=2560)
    ys_task = [1, 0, 0]
    coefficient = [1.0, 1.0, 1.0]
    ys_emotion = [4]
    model = EcgNetwork(3, 5)
    model.is_pretext = True
    out_list = model(zero_input)
    loss_task = nn.BCELoss()
    total_loss_task = sum([loss_task(o, u.y_to_torch([y], shape=(1, 1))) * c for o, y, c in zip(out_list, ys_task, coefficient)])
    total_loss_task.backward()
    model.is_pretext = False
    out = model(zero_input)
    loss_emotion = nn.NLLLoss()
    total_loss_emotion = loss_emotion(out, torch.tensor(ys_emotion))
    total_loss_emotion.backward()


def test_augmentations():

    def print_teaser(aug_name, signal):
        print(f'\nstarting test for {aug_name}:\n-----------------------------')
        if signal is not None:
            print(f'signal: {signal}')

    def print_footer(aug_name):
        print(f'-----------------------------\nend {aug_name} test\n')

    signal = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 0.0, 0.3])

    print_teaser('add noise', signal)
    noise = aug.add_noise(signal, 1)
    assert signal.shape == noise.shape
    print(f'noise: {noise}')
    print_footer('add noise')

    print_teaser('scale', signal)
    scaled = aug.scale(signal, 2)
    print(f'scaled: {scaled}')
    assert (scaled == signal * 2).all()
    print_footer('scale')

    print_teaser('negate', signal)
    negated = aug.negate(signal)
    print(f'negated: {negated}')
    assert (negated == signal * -1).all()
    print_footer('negate')

    print_teaser('temporary invert', signal)
    inverted = aug.temp_invert(signal)
    print(f't_inv: {inverted}')
    assert signal.shape == inverted.shape
    assert (signal == inverted[::-1]).all()
    print_footer('temporary invert')

    print_teaser('permutate', signal)
    permutated = aug.permuatate(signal, 3)
    print(f'permutated: {permutated}')
    assert signal.shape == permutated.shape
    print_footer('permutate')

    print_teaser('time warp', signal)
    n_sections = 3
    k = 3
    t_warp = aug.time_warp(signal, n_sections, k)
    print(f'params: n_sections={n_sections}, k={k}')
    print(f't_warp: {t_warp}')
    assert signal.shape == t_warp.shape
    print_footer('time warp')


