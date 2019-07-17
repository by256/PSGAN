import numpy as np


def sample_Z_l(batch_size, d_l, L, M):
    outputs = []

    for _ in range(batch_size):
        Z_l = np.zeros(shape=(d_l, L, M))

        for i in range(L):
            for j in range(M):
                sample = np.random.uniform(-1.0, 1.0, size=(d_l))
                Z_l[:, i, j] = sample
        outputs.append(Z_l)

    return np.array(outputs)

def sample_Z_g(batch_size, d_g, L, M):
    outputs = []
    for _ in range(batch_size):
        z_g = np.random.uniform(-1.0, 1.0, size=(d_g, 1, 1))
        Z_g = np.tile(z_g, (1, L, M))
        outputs.append(Z_g)

    return np.array(outputs)

def interpolated_Z_g(batch_size, d_g, L, M, rate):

    Z_g = np.zeros(shape=(batch_size, d_g, L, M))

    top_left = np.random.uniform(-1, 1, (batch_size, d_g))
    top_right = np.random.uniform(-1, 1, (batch_size, d_g))
    bottom_left = np.random.uniform(-1, 1, (batch_size, d_g))
    bottom_right = np.random.uniform(-1, 1, (batch_size, d_g))

    for m in range(M):
        for l in range(L):
            z_g = (1-m/rate)*(1-l/rate)*top_left + (1-m/rate)*(l/rate)*top_right + (m/rate)*(1-l/rate)*bottom_left + (m/rate)*(l/rate)*bottom_right
            Z_g[:, :, l, m] = z_g

    return Z_g
