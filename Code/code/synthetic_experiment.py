import numpy as np
import argparse
import os
import torch
from scipy.stats import norm
from tqdm import tqdm

def func1(x):
    return -10 * np.exp(2 * x)

class Experiment:
    def __init__(self, n, q_dist_info, p_dist_info, f, n_exp):
        self.n = n
        self.Q = q_dist_info
        self.P = p_dist_info
        self.f = f
        self.n_exps = n_exp
        self.pm_lambdas = [0, 0.1, 0.3, 0.5, 0.8]
        self.lse_lambdas = [0.01, 0.1, 1, 5]

    def print_results(self, errors):
        errors = np.array(errors)
        print(np.mean(errors, axis=0))
        print(np.std(errors, axis=0))

    def prepare_samples(self):
        samples = np.random.normal(self.Q[0], np.sqrt(self.Q[1]), size=(self.n, 1))
        q = norm.pdf(samples, self.Q[0], np.sqrt(self.Q[1]))
        p = norm.pdf(samples, self.P[0], np.sqrt(self.P[1]))
        func_values = self.f(samples)
        return q, p, func_values

    def calculate_pm_expected_value(self, pm_lambda, q, p, func_values):
        w =  p / q
        power_mean_w = w / (1 - pm_lambda + (pm_lambda * w))
        return np.mean(func_values * power_mean_w)

    def calculate_lse_expected_value(self, lse_lambda, q, p, func_values):
        result = lse_lambda * (func_values) * (p / q)
        result = np.exp(result)
        result = np.log(np.mean(result))
        return ((1 / lse_lambda) * result)

    def run_pm_experiments(self, correct_expected_value, q, p, func_values):
        errors = []
        for pm_lambda in self.pm_lambdas:
            pm_expected_value = self.calculate_pm_expected_value(pm_lambda, q, p, func_values)
            errors.append(np.abs(pm_expected_value - correct_expected_value))
        return errors

    def run_lse_experiments(self, correct_expected_value, q, p, func_values):
        errors = []
        for lse_lambda in self.lse_lambdas:
            lse_expected_value = self.calculate_lse_expected_value(lse_lambda, q, p, func_values)
            errors.append(np.abs(lse_expected_value - correct_expected_value))
        return errors

    def run(self):
        correct_expected_value = -10 * np.exp(2 * self.P[0] + 2 * self.P[1] * self.P[1])
        pm_errors = []
        lse_errors = []

        for epoch in tqdm(range(self.n_exps)):
            q, p, func_values = self.prepare_samples()
            pm_errors.append(self.run_pm_experiments(correct_expected_value, q, p, func_values))
            lse_errors.append(self.run_lse_experiments(correct_expected_value, q, p, func_values))
        self.print_results(pm_errors)
        print("----------------------------")
        self.print_results(lse_errors)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True, help="Number of Samples")
parser.add_argument("--q_dist", type=float, nargs=2, required=True, help="Q Distribution mean and variance")
parser.add_argument("--p_dist", type=float, nargs=2, required=True, help="P Distribution mean and variance")
parser.add_argument("--n_exp", type=int, required=True)
args = parser.parse_args()

new_exp = Experiment(args.n, args.q_dist, args.p_dist, func1, args.n_exp)
new_exp.run()




