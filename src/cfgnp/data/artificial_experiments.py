import torch
import numpy as np
import argparse
import pickle
from scipy import stats as sts
import os

def backdoor_dgp(N=1000, P=10, rng=1):
    # Set seed
    torch.manual_seed(rng)
    torch.cuda.manual_seed(rng)
    np.random.seed(rng)


    # Generate X with a little bit of correlation b/w the continuous variables (i.e. 0.3 Pearson coeffincient c.a.)
    X = np.random.uniform(-3., 3, (N, P))
    X_ones = np.c_[np.ones(N), X]

    # Generate A
    coef1 = np.zeros(P+1)
    coef2 = np.zeros(P+1)
    coef3 = np.zeros(P+1)
    coef4 = np.zeros(P+1)

    coef4[range(4)] = np.array([2, 1.5, 0.5, 0.2])
    coef3[range(4)] = np.array([1.5, 0.6, 0.3, 0.3])
    coef2[range(2, 5)] = np.array([1, 0.8, 0.2])
    coef1[range(4)] = np.array([-1, -0.8, -0.1, -0.1])

    und_lin = np.c_[np.exp(X_ones @ coef1), np.exp(X_ones @ coef2),
                    np.exp(X_ones @ coef3), np.exp(X_ones @ coef4)]

    for i in range(N):
        und_lin[i, :] /= und_lin[i, :].sum(axis=0)

    np.random.multinomial(1, und_lin[0, :])

    A = np.array([np.random.multinomial(1, und_lin[i, :]) for i in range(N)])
    # Generate Y
    Y_true = np.zeros((N, 4))
    Y_true[:, 0] = 3 + 0.4 * X[:, 0]*X[:, 1] - 0.3 * X[:, 2] ** 2 + 0.2 * np.exp(X[:, 3]) + 0.6 * np.sin(X[:, 4])
    Y_true[:, 1] = -1 + Y_true[:, 0] + 0.1 * X[:, 5]
    Y_true[:, 2] = 1 + Y_true[:, 0] + 0.3 * X[:, 5]
    Y_true[:, 3] = 0.5 + Y_true[:, 0] + 0.5 * X[:, 6]

    # Generate C
    C_true = np.zeros((N, 4))
    C_true[:, 0] = 1 + 0.2 * X[:, 0]*X[:, 1] - 0.2 * X[:, 2] ** 2 + 0.1 * np.exp(X[:, 3])
    C_true[:, 1] = -2 + C_true[:, 0] + 0.2 * X[:, 5]
    C_true[:, 2] = 2 + C_true[:, 0] + 0.4 * X[:, 5]
    C_true[:, 3] = 1 + C_true[:, 0] + 0.5 * X[:, 6]

    sigma_Y = 0.5
    sigma_C = 0.5
    Y_obs = (Y_true + sts.norm.rvs(0, sigma_Y, (N, 4))) * A
    C_obs = (C_true + sts.norm.rvs(0, sigma_C, (N, 4))) * A

    Y = Y_obs[Y_obs != 0]
    C = C_obs[C_obs != 0]

    A_ = np.zeros(N)

    for i in range(4):
        A_[A[:, i] == 1] = i

    return X, A_, Y, C, Y_true, C_true

def generate_cf_ellipse(num_samples, num_obs, seed=42, dir="./data/ellipse_data"):
    if os.path.exists(f'{dir}/dataset_{seed}.pt'):
        samples = torch.load(f'{dir}/dataset_{seed}.pt')
        return samples

    np.random.seed(seed)

    z = np.random.uniform(low=-0.5, high=0.5, size=(num_samples,)) # Z
    epsilon_t = np.random.normal(loc=0.0, scale=1.0, size=(num_samples,))
    epsilon_r = np.random.exponential(scale=1.0, size=(num_samples,))
    epsilon_b = np.random.beta(a=1.0, b=1.0, size=(num_samples,))
    weights = np.random.uniform(low=1.0, high=2.0, size=(3,))
    biases = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    t = (weights[0] * z + biases[0] + epsilon_t) % (2 * np.pi) # X
    r = 1 + np.multiply(np.exp(weights[1] * z + biases[1]), epsilon_r)
    b = np.exp(weights[2] * z + biases[2]) + epsilon_b # U_0

    t1 = (weights[0] * z + biases[0] + epsilon_t) % (2 * np.pi) # X
    r1 = 1 + np.exp(weights[1] * z + biases[1]) * epsilon_r
    y1 = b * (2 + np.sin(t1)) # V_0
    x1 = (r1 * b) * (2 + np.cos(t1)) # V_1

    # count_data
    eps_t2 = np.random.normal(loc=0.0, scale=1.0, size=(num_samples,))
    eps_r2 = np.random.exponential(scale=1.0, size=(num_samples,))

    t2 = (weights[0] * z + biases[0] + eps_t2) % (2 * np.pi) # X_prime
    r2 = 1 + np.exp(weights[1] * z + biases[1]) * eps_r2
    y2 = b * (2 + np.sin(t2)) # V_0_prime
    x2 = (r2 * b) * (2 + np.cos(t2)) # V_1_prime

    sample_orig = np.stack([z, b, t1, y1, x1], axis=-1)[..., np.newaxis]
    sample_int = np.zeros_like(sample_orig)
    sample_int[:, 2] = t2[..., np.newaxis]

    # oberservational data
    z_obs = np.random.uniform(low=-0.5, high=0.5, size=(num_samples * num_obs,)) # Z
    epsilon_t = np.random.normal(loc=0.0, scale=1.0, size=(num_samples * num_obs,))
    epsilon_r = np.random.exponential(scale=1.0, size=(num_samples * num_obs,))
    epsilon_b = np.random.beta(a=1.0, b=1.0, size=(num_samples * num_obs,))
    weights = np.random.uniform(low=1.0, high=2.0, size=(3,))
    biases = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    t_obs = (weights[0] * z_obs + biases[0] + epsilon_t) % (2 * np.pi) # X
    r_obs = 1 + np.multiply(np.exp(weights[1] * z_obs + biases[1]), epsilon_r)
    b_obs = np.exp(weights[2] * z_obs + biases[2]) + epsilon_b # U_0
    y_obs = np.multiply(b_obs, 2 + np.sin(t_obs)) # V_0
    x_obs = np.multiply(np.multiply(r_obs, b_obs), 2 + np.cos(t_obs)) # V_1
    sample_obs = np.stack([z_obs, t_obs, b_obs, y_obs, x_obs], axis=-1)
    sample_obs = sample_obs.reshape((num_samples, num_obs, 5, 1))

    sample_target = np.stack([z, b, t2, y2, x2], axis=-1)[..., np.newaxis]
    interventional_indices = 2 * np.ones((num_samples, 1, 1), dtype=int)
    samples_reshaped = []
    for i in range(num_samples):
        samples_reshaped.append(((torch.Tensor(sample_int[i]).unsqueeze(0), torch.Tensor(interventional_indices[i]).unsqueeze(0), torch.Tensor(sample_orig[i]).unsqueeze(0), torch.Tensor(sample_obs[i]).unsqueeze(0)), torch.Tensor(sample_target[i]).unsqueeze(0)))
    os.makedirs(dir, exist_ok=True)
    torch.save(samples_reshaped, f'{dir}/dataset_{seed}.pt')
    return samples_reshaped