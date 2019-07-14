import numpy as np


def sample_Z_l(batch_size, L, M, d_l):
    outputs = []

    for _ in range(batch_size):
        Z_l = np.zeros(shape=(L, M, d_l))

        for i in range(L):
            for j in range(M):
                sample = np.random.uniform(size=(1, 1, d_l))
                Z_l[i, j, :] = sample
        outputs.append(Z_l)

    return np.array(outputs)

def sample_z_g(batch_size, d_g):
    outputs = []
    for _ in range(batch_size):
        Z_g = np.random.uniform(size=(d_g,))
        outputs.append(Z_g)

    return np.array(outputs)


