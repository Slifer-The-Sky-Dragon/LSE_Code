import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from utils import *

# ACTION_SIZE = 10


class RewardLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params

    def forward(self, output, action, delta):
        return F.binary_cross_entropy(output[range(action.size(0)), action], delta)


class CustomLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lamda = hyper_params["lamda"]

    def forward(self, output, action, delta, prop):
        risk = -delta
        loss = (risk - self.lamda) * (output[range(action.size(0)), action] / prop)
        return torch.mean(loss)


class IPS_C_LOSS(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.c = self.hyper_params["ips_c"]
        self.lamda = hyper_params["lamda"]

    def forward(self, output, action, delta, prop):
        risk = -delta
        loss = (risk - self.lamda) * (
            output[range(action.size(0)), action] / (prop + self.c)
        )
        return torch.mean(loss)


class CustomLossRec(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lamda = hyper_params["lamda"]

    def forward(self, output, delta, prop):
        risk = -delta

        loss = (risk - self.lamda) * (output / prop)

        return torch.mean(loss)


def SecondMomentLoss(output, action, prop, action_size=10):
    h_scores = output[range(action.size(0)), action]
    all_values = (h_scores / (prop + 1e-8)) ** 2
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


class LSE_Loss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lse_lamda = hyper_params["lse_lamda"]
        self.lamda = hyper_params["lamda"]

    def forward(self, output, action, delta, prop):
        risk = -delta
        loss = (
            self.lse_lamda
            * (risk - self.lamda)
            * (output[range(action.size(0)), action] / prop)
        )
        max_loss = torch.amax(loss, dim=-1, keepdim=True)
        loss = torch.exp(loss - max_loss)
        loss = torch.log(torch.mean(loss)) + max_loss
        return (1 / self.lse_lamda) * loss


class LSE_LossRec(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.lse_lamda = hyper_params["lse_lamda"]
        self.lamda = hyper_params["lamda"]

    def forward(self, output, delta, prop):
        risk = -delta
        loss = self.lse_lamda * (risk - self.lamda) * (output / prop)
        max_loss = torch.amax(loss, dim=-1, keepdim=True)
        loss = torch.exp(loss - max_loss)
        loss = torch.log(torch.mean(loss)) + max_loss
        return (1 / self.lse_lamda) * loss


class PowerMeanLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.power_mean_lamda = hyper_params["power_mean_lamda"]
        self.lamda = hyper_params["lamda"]

    def forward(self, output, action, delta, prop):
        risk = -delta
        w = output[range(action.size(0)), action] / prop
        power_mean_w = w / (1 - self.power_mean_lamda + (self.power_mean_lamda * w))
        loss = (risk - self.lamda) * power_mean_w
        return torch.mean(loss)


class PowerMeanLossRec(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.power_mean_lamda = hyper_params["power_mean_lamda"]
        self.lamda = hyper_params["lamda"]

    def forward(self, output, delta, prop):
        risk = -delta
        w = output / prop
        power_mean_w = w / (1 - self.power_mean_lamda + (self.power_mean_lamda * w))
        loss = (risk - self.lamda) * power_mean_w
        return torch.mean(loss)


class ExponentialSmoothingLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.alpha = hyper_params["exs_alpha"]
        self.lamda = self.lamda = hyper_params["lamda"]

    def forward(self, output, action, delta, prop):
        risk = -delta
        w = output[range(action.size(0)), action] / torch.pow(prop, self.alpha)
        loss = (risk - self.lamda) * w
        return torch.mean(loss)


class ExponentialSmoothingLossRec(torch.nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        self.alpha = hyper_params["exs_alpha"]
        self.lamda = self.lamda = hyper_params["lamda"]

    def forward(self, output, delta, prop):
        risk = -delta
        w = output / torch.pow(prop, self.alpha)
        loss = (risk - self.lamda) * w
        return torch.mean(loss)


# def KLLoss(output, action, prop):
#     h_scores = output[range(action.size(0)), action]
#     return torch.sum(h_scores * torch.log(H_scores / prop + 1e-8))


def KLLoss(output, action, prop, action_size=10):
    h_scores = output[range(action.size(0)), action]
    all_values = h_scores * torch.log(h_scores / prop + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


def KLLossRec(output, action, prop):
    action_size = torch.max(action) + 1
    h_scores = output
    all_values = h_scores * torch.log(h_scores / prop + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


def KLLossRev(output, action, prop, action_size=10):
    h_scores = output[range(action.size(0)), action]
    all_values = prop * torch.log(prop / (h_scores + 1e-8) + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


def KLLossRevRec(output, action, prop):
    action_size = torch.max(action) + 1
    h_scores = output
    all_values = prop * torch.log(prop / (h_scores + 1e-8) + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)


def AlphaRenyiLoss(output, action, prop, num_classes, hyper_params):
    alpha = hyper_params["ar_alpha"]
    beta = hyper_params["ar_beta"]
    type = hyper_params["ar_type"]

    if abs(alpha - 1) < 0.001:
        return KLLoss(output=output, action=action, prop=prop, action_size=num_classes)

    if type == 1:
        if alpha > 0 and alpha < 1:
            w = torch.pow(prop, alpha) * torch.pow(
                output[range(action.size(0)), action], 1 - alpha
            )
        else:
            w = torch.pow(prop, alpha) / torch.pow(
                output[range(action.size(0)), action], alpha - 1
            )
    else:
        if alpha > 0 and alpha < 1:
            w = torch.pow(output[range(action.size(0)), action], alpha) * torch.pow(
                prop, 1 - alpha
            )
        else:
            w = torch.pow(output[range(action.size(0)), action], alpha) / torch.pow(
                prop, alpha - 1
            )

    w = torch.reshape(w, (len(w), 1))
    one_hot_y = F.one_hot(action, num_classes=num_classes).float()
    action_sum = torch.mm(one_hot_y.T, w)
    action_count = torch.reshape(one_hot_y.sum(dim=0), (num_classes, 1))
    action_mean = action_sum / (action_count + 1e-8)

    all_action_mean = torch.sum(action_mean[action_count != 0])

    return (beta / (alpha - 1)) * torch.log(all_action_mean)


def AlphaRenyiLossRec(output, action, prop, hyper_params):
    alpha = hyper_params["ar_alpha"]
    beta = hyper_params["ar_beta"]
    type = hyper_params["ar_type"]
    num_classes = torch.max(action) + 1

    if abs(alpha - 1) < 0.001:
        return KLLossRec(output=output, prop=prop, action_size=num_classes)

    if type == 1:
        if alpha > 0 and alpha < 1:
            w = torch.pow(prop, alpha) * torch.pow(output, 1 - alpha)
        else:
            w = torch.pow(prop, alpha) / torch.pow(output, alpha - 1)
    else:
        if alpha > 0 and alpha < 1:
            w = torch.pow(output, alpha) * torch.pow(prop, 1 - alpha)
        else:
            w = torch.pow(output, alpha) / torch.pow(prop, alpha - 1)

    w = torch.reshape(w, (len(w), 1))
    one_hot_y = F.one_hot(action, num_classes=num_classes).float()
    action_sum = torch.mm(one_hot_y.T, w)
    action_count = torch.reshape(one_hot_y.sum(dim=0), (num_classes, 1))
    action_mean = action_sum / (action_count + 1e-8)

    all_action_mean = torch.sum(action_mean[action_count != 0])

    return (beta / (alpha - 1)) * torch.log(all_action_mean)


def SupKLLoss(output, action, delta, prop, eps, action_size=10):
    h_scores = output[range(action.size(0)), action]
    all_values = h_scores * torch.log(h_scores * (delta + eps) / prop + 1e-8)
    index = action
    src = all_values
    out = torch.scatter_reduce(
        torch.zeros(action_size).to(output.device),
        dim=0,
        index=index,
        src=src,
        reduce="mean",
    )
    return torch.sum(out)
