import numpy as np
from scipy import sparse

from ._checks import NUMERIC_KINDS, real_option
from .errors import PPIxGPNError


def ppi_network(network_data, threshold=0.4):
    threshold = real_option(threshold, "threshold", minimum=0, open_minimum=True)
    is_sparse = sparse.issparse(network_data)
    rows, columns, score, num_protein = (_sparse_links if is_sparse else _dense_links)(network_data, threshold)
    weight = 1 / (1 + np.exp(-_standardize(score)))
    if not is_sparse:
        adjacency = np.zeros((num_protein, num_protein))
        adjacency[rows, columns] = weight
        scale = _inverse_root(adjacency.sum(axis=1))
        return np.eye(num_protein) - (scale[:, None] * adjacency) * scale[None, :]
    scale = _inverse_root(np.bincount(rows, weights=weight, minlength=num_protein))
    coupling = sparse.coo_array((-(scale[rows] * weight) * scale[columns], (rows, columns)),
                                shape=(num_protein, num_protein))
    laplacian = sparse.csc_array(sparse.identity(num_protein, format="csc") + coupling.tocsc())
    laplacian.eliminate_zeros()
    return sparse.csc_matrix(laplacian) if sparse.isspmatrix(network_data) else laplacian


def _dense_links(network_data, threshold):
    network = np.asarray(network_data)
    _check_network(network.dtype, network.shape, network)
    network = network.astype(np.float64, copy=False)
    columns, rows = np.nonzero(network.T >= threshold)
    return rows, columns, network[rows, columns], network.shape[0]


def _sparse_links(network_data, threshold):
    entries = sparse.coo_array(network_data)
    entries.sum_duplicates()
    _check_network(entries.dtype, entries.shape, entries.data)
    values = entries.data.astype(np.float64)
    linked = values >= threshold
    rows, columns, values = entries.row[linked].astype(np.intp), entries.col[linked].astype(np.intp), values[linked]
    order = np.lexsort((rows, columns))
    return rows[order], columns[order], values[order], entries.shape[0]


def _check_network(dtype, shape, values):
    if dtype.kind not in NUMERIC_KINDS or len(shape) != 2 or shape[0] != shape[1]:
        raise PPIxGPNError("PPIxGPN:InvalidNetwork", "network_data must be a square protein-by-protein matrix.")
    if not np.isfinite(values).all():
        raise PPIxGPNError("PPIxGPN:InvalidNetwork", "network_data must contain only finite values.")


def _standardize(score):
    if score.size == 0 or np.all(score == score[0]):
        return np.zeros_like(score)
    return (score - score.mean()) / score.std(ddof=1)


def _inverse_root(degree):
    scale = np.zeros_like(degree)
    np.divide(1, np.sqrt(degree), out=scale, where=degree > 0)
    return scale
