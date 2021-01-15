import numpy as np


def golub(a, b):
    """
    LSP-solver based on the Golub's method
    :param a: M x N - matrix A
    :param b: M - vector b
    :return: N - vector x
    """
    # QR decomposition of A
    [q, r] = np.linalg.qr(a)

    # compute new b
    bnew = np.matmul(q.T, b)

    # get size of A
    [m, n] = a.shape
    # compute solution
    x = np.linalg.solve(r[0:n, :], bnew[0:n])

    # compute residual
    residual = np.sqrt(np.sum((np.matmul(a, x) - b) ** 2))

    return x, residual


def solve_lsp(a, b):
    """
    Method to solve the least squares problem for Ax=b
    :param a: M x N - matrix A
    :param b: M - vector b
    :return: N - vector x
    """
    return golub(a, b)
