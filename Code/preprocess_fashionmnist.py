import numpy as np
import pickle
from tqdm import tqdm

avg_num_zeros = 0.0
from code.model_h0 import ModelCifar
from functools import partial
from code.hyper_params import load_hyper_params
import argparse
import torch
import os
from torchvision.datasets import FashionMNIST

np.random.seed(2023)
torch.manual_seed(2023)


def one_hot(arr):
    new = []
    for i in tqdm(range(len(arr))):
        temp = np.zeros(10)
        temp[arr[i]] = 1.0
        new.append(temp)
    return np.array(new)


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_probs(label):
    probs = np.power(np.add(list(range(10)), 1), 0.5)
    probs[label] = probs[label] * 10

    probs = probs / probs.sum()

    return probs


def get_biased_probs(label):
    probs = np.power(np.add(list(range(10)), 1), 0.5)
    probs[label] = probs[label] * 1.5

    probs = probs / probs.sum()

    return probs


def get_wrong_probs(label):
    probs = np.ones(10)
    probs[(label + 1) % 10] *= 10

    probs = probs / probs.sum()

    return probs


def get_uniform_probs(label):
    probs = np.ones(10)

    probs = probs / probs.sum()

    return probs


def get_model_probs(model, image):
    with torch.no_grad():
        image = torch.tensor(image.reshape(1, 3, 32, 32)).float()
        probs = (
            torch.softmax(model(image.to(device)), dim=-1)
            .cpu()
            .numpy()
            .squeeze(0)
            .astype(np.float32)
        )
    return probs


def get_model_probs_we(image, model, eps):
    with torch.no_grad():
        image = torch.tensor(image.reshape(1, 3, 32, 32)).float()
        probs = (
            torch.softmax(model(image.to(device)), dim=-1)
            .cpu()
            .numpy()
            .squeeze(0)
            .astype(np.float32)
        )
    probs[probs < eps] = eps
    probs /= np.sum(probs)
    return probs

x_train, x_val, x_test = [], [], []
y_train, y_val, y_test = [], [], []

train_dataset = FashionMNIST(root='data/fmnist/', train=True)
test_dataset = FashionMNIST(root='data/fmnist/', train=False)
N, M = len(train_dataset), len(test_dataset)
print("Len Train =", N)
print("Len Test =", M)

# Train
for i in range(N):
    image, label = train_dataset[i]
    image = np.array(image).reshape(28 * 28)
    x_train.append(image)
    y_train.append(label)
x_train = np.stack(x_train)
y_train = np.stack(y_train)

# Test
for i in range(M):
    image, label = test_dataset[i]
    image = np.array(image).reshape(28 * 28)
    x_test.append(image)
    y_test.append(label)
x_test = np.stack(x_test)
y_test = np.stack(y_test)

# Normalize X data
x_train = x_train.astype(float) / 255.0
x_test = x_test.astype(float) / 255.0

# One hot the rewards
y_train = one_hot(y_train)
y_test = one_hot(y_test)

# Shuffle the dataset once
indices = np.arange(len(x_train))
np.random.shuffle(indices)
assert len(x_train) == len(y_train)
x_train = x_train[indices]
y_train = y_train[indices]
print(x_train.shape)
N = len(x_train)

prop_train = np.zeros((N, 10))
actions_train = np.zeros((N, 1))
labeled_train = np.ones((N, 1))

train_and_val = np.concatenate([x_train, y_train, prop_train, actions_train, labeled_train], axis=1)

idx = list(range(N))
idx = np.random.permutation(idx)
train = train_and_val[idx[:50000]]
val = train_and_val[idx[50000:]]

prop_test = np.zeros((M, 10))
actions_test = np.zeros((M, 1))
labeled_test = np.ones((M, 1))

test = np.concatenate([x_test, y_test, prop_test, actions_test, labeled_test], axis=1)
print("Train size:", train.shape)
print("Val size:", val.shape)
print("Test size:", test.shape)
np.save("data/fmnist/base/train.npy", train)
np.save("data/fmnist/base/val.npy", val)
np.save("data/fmnist/base/test.npy", test)
