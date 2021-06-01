from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn


class EcgCNN(nn.Module):
    def __init__(self):
        super(EcgCNN, self).__init__()

        self.conv_1 = nn.Conv1d(1, 32, 32)
        self.conv_2 = nn.Conv1d(32, 32, 32)

        self.pool_1 = nn.MaxPool1d(8, 2)

        self.conv_3 = nn.Conv1d(32, 64, 16)
        self.conv_4 = nn.Conv1d(64, 64, 16)

        self.pool_2 = nn.MaxPool1d(8, 2)

        self.conv_5 = nn.Conv1d( 64, 128, 8)
        self.conv_6 = nn.Conv1d(128, 128, 8)

        self.pool_3 = nn.MaxPool1d(635)

        self.layers = [self.conv_1, self.conv_2, self.pool_1,
                       self.conv_3, self.conv_4, self.pool_2,
                       self.conv_5, self.conv_6, self.pool_3]

        self.debug_dim = False

    def forward(self, time_series):
        def dd(x):
            if self.debug_dim:
                print(x.shape)

        x = time_series

        dd(x)
        x = F.pad(x, (16, 15, 0, 0))
        x = self.conv_1(x)
        x = F.leaky_relu(x)
        dd(x)
        x = F.pad(x, (16, 15, 0, 0))
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        dd(x)

        x = self.pool_1(x)
        dd(x)

        x = F.pad(x, (8, 7, 0, 0))
        x = self.conv_3(x)
        x = F.leaky_relu(x)
        dd(x)
        x = F.pad(x, (8, 7, 0, 0))
        x = self.conv_4(x)
        x = F.leaky_relu(x)
        dd(x)

        x = self.pool_2(x)
        dd(x)

        x = F.pad(x, (4, 3, 0, 0))
        x = self.conv_5(x)
        x = F.leaky_relu(x)
        dd(x)
        x = F.pad(x, (4, 3, 0, 0))
        x = self.conv_6(x)
        x = F.leaky_relu(x)
        dd(x)

        x = self.pool_3(x)
        dd(x)

        x = x.squeeze(dim=2)
        dd(x)

        return x


class EcgHead(nn.Module):
    def __init__(self, n_out=1, drop_rate=0.6):
        super(EcgHead, self).__init__()

        self.head_1 = nn.Linear(128, 128)
        self.head_2 = nn.Linear(128, 128)
        self.head_3 = nn.Linear(128, n_out)
        self.dropout = nn.Dropout(drop_rate)
        # we changed to BCEWithLogitLoss and CrossEntropyLoss, as they perform better, so the networks only output logits
        self.out_activation = None  # torch.sigmoid if n_out == 1 else None

        self.debug_dim = False
        self.debug_values = False

    def forward(self, x):
        if self.debug_values:
            print('input', x)
        def ff_block(x, head):
            x = head(x)
            x = F.leaky_relu(x)
            x = self.dropout(x)
            return x
        x = ff_block(x, self.head_1)
        x = ff_block(x, self.head_2)
        x = self.head_3(x)
        if self.debug_dim:
            print('head before activation', x.shape)
        if self.debug_values:
            print('before last activation', x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        if self.debug_values:
            print('after last activation', x)
        if self.debug_dim:
            print('head after activation', x.shape)
        return x

class EcgAmigosHead(nn.Module):
    def __init__(self, n_out=1, drop_rate=0.6):
        super(EcgAmigosHead, self).__init__()

        self.head_1 = nn.Linear(128, 512)
        self.head_2 = nn.Linear(512, 512)
        self.head_3 = nn.Linear(512, 512)
        self.head_4 = nn.Linear(512, n_out)
        self.dropout = nn.Dropout(drop_rate)
        # we changed to BCEWithLogitLoss and CrossEntropyLoss, as they perform better, so the networks only output logits
        self.out_activation = None  # torch.sigmoid if n_out == 1 else None

        self.debug_dim = False
        self.debug_values = False

    def forward(self, x):
        if self.debug_values:
            print('input', x)
        def ff_block(x, head):
            x = head(x)
            x = F.relu(x)
            x = self.dropout(x)
            return x
        x = ff_block(x, self.head_1)
        x = ff_block(x, self.head_2)
        x = ff_block(x, self.head_3)
        x = self.head_4(x)
        x = self.dropout(x)
        if self.debug_dim:
            print('head before activation', x.shape)
        if self.debug_values:
            print('before last activation', x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        if self.debug_values:
            print('after last activation', x)
        if self.debug_dim:
            print('head after activation', x.shape)
        return x


class EcgNetwork(nn.Module):
    def __init__(self, n_task_heads, n_emotions):
        super(EcgNetwork, self).__init__()

        self.cnn = EcgCNN()

        self.task_heads = [EcgHead() for _ in range(n_task_heads)]
        self.tasks_out_activation = nn.LogSoftmax(dim=1)

        self.emotion_head = EcgHead(n_out=n_emotions)

        self.is_pretext = True

    def forward(self, x):
        embedding = self.cnn(x)

        if self.is_pretext:
            x_list = [th(embedding) for th in self.task_heads]
            out_stacked = torch.stack(x_list)
            return out_stacked, embedding
        else:
            x = self.emotion_head(embedding)
            return x, embedding

    def _apply(self, fn):
        self.task_heads = [fn(th) for th in self.task_heads]
        self.emotion_head = fn(self.emotion_head)
        self.cnn = fn(self.cnn)
        return self


class AvaragePretextLoss(nn.Module):
    def __init__(self, per_task_criterion, coefficients):
        super(AvaragePretextLoss, self).__init__()
        self.per_task_criterion = per_task_criterion
        self.coefficients = coefficients

    def forward(self, targets, labels):
        # matrix of (batch_size, n_tasks, 1)
        total_loss = sum(
            sum(
                self.per_task_criterion(targets, labels)
            )
        ) / (targets.shape[0] * targets.shape[1])
        return total_loss

    def _apply(self, fn):
        self.per_task_criterion = fn(self.per_task_criterion)
        self.coefficients = fn(self.coefficients)
        return self

    # def forward(self, tasks_output, labels):
    #     total_loss = sum(
    #         [self.per_task_criterion(o, y) * c for o, y, c in zip(tasks_output, labels, self.coefficients)])
    #     return total_loss


def labels_to_vec(labels, n_tasks, debug_ltv=False):
    binary_matrix = torch.zeros((len(labels), n_tasks))
    for i in range(n_tasks):
        l_vec = (labels == i).int().float()
        binary_matrix[:, i] = l_vec
    if debug_ltv:
        print(binary_matrix)
        print(binary_matrix.shape)
    return binary_matrix
# def labels_to_vec(labels, n_tasks, debug_ltv=False):
#     def p(*args, **kwargs):
#         if debug_ltv:
#             print(args)
#     all_labels = torch.zeros((len(labels[0]), n_tasks))
#     p(n_tasks)
#     p(all_labels.shape)
#     for l in labels:
#         for i in range(n_tasks):
#             batch_i = (l == i).int().float()
#             #p(l)
#             #p(i)
#             p(batch_i)
#             p('bi size ', batch_i.shape)
#             all_labels[:,i] += batch_i
#     return all_labels