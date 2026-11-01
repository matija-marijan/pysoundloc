import numpy as np
import torch

from scipy.interpolate import interp1d as np_interp1d

from .gcc import Gcc #, gcc_phat_batch
from .tdoa import compute_theoretical_tdoa_grids
from .utils.torchinterp1d import interp1d as torch_interp1d


def compute_pairwise_srp_grids(mic_0_signals, mic_1_signals, sr,
                               mic_0_coords, mic_1_coords,
                               room_dims, n_grid_points=25,
                               mode="max_correlation_neighbours",
                               n_correlation_neighbours=10,
                               phase_transform=True,
                               gcc_mode="stft",
                               n_dft=8192):
    
    # 0. Infer which library to use (numpy or torch)
    if isinstance(mic_0_signals, torch.Tensor):
        lib = torch
    elif isinstance(mic_0_signals, np.ndarray):
        lib = np
    else:
        raise ValueError("Signals must be either torch tensors or numpy arrays")
    
    batch_size = room_dims.shape[0]

    # 1. Compute the theoretical time delay for a pair of microphones and the candidate points in a grid.
    theoretical_tdoa_grids = compute_theoretical_tdoa_grids(mic_0_coords, mic_1_coords, room_dims, n_grid_points)

    # 2. Compute the GCC-PHAT between the pairs of microphone signals
    gcc_func = Gcc(sr, abs=True, phase_transform=phase_transform,
                   mode=gcc_mode, n_dft=n_dft)
    
    # GCC only implemented in torch, move arrays to tensors
    if lib == np:
        mic_0_signals = torch.from_numpy(mic_0_signals)
        mic_1_signals = torch.from_numpy(mic_1_signals)   
    cc_vectors, cc_idxs = gcc_func(mic_0_signals, mic_1_signals)
    if lib == torch:
        cc_idxs = cc_idxs.to(cc_vectors.device)
    if lib == np:
        cc_vectors, cc_idxs = cc_vectors.numpy(), cc_idxs.numpy()

    #cc_vectors, cc_idxs = gcc_phat_batch(mic_0_signals, mic_1_signals, sr, abs=True)

    if mode == "max_correlation_neighbours":
        srp_grids = _n_corr_neighbours_srp_batch(
            cc_idxs,
            cc_vectors,
            theoretical_tdoa_grids,
            n_correlation_neighbours,
            lib,
        )
    elif mode == "interpolate":
        srp_grids = lib.stack([
            _interpolated_srp(cc_idxs, cc_vectors[i], theoretical_tdoa_grids[i])
            for i in range(batch_size)
        ])
    else:
        raise ValueError(f"Unknown SRP mode: {mode!r}")
    
    return srp_grids


def _n_corr_neighbours_srp(gcc_idxs, gcc_values, theoretical_tdoa_grid, n_correlation_neighbours, lib):
    if lib == torch:
        max_func = lambda x: torch.max(x, dim=0)[0]
    elif lib == np:
        max_func = lambda x: np.max(x, axis=0)
    else:
        raise ValueError("'lib' must be numpy or torch")

    # 1. Get the index of the theoretical time delay that is the closest to integer sample delays by gcc
    idxs_grid = lib.searchsorted(gcc_idxs, theoretical_tdoa_grid)

    # 2. Get the 'n_correlation_neighbours' central values in the GCC-PHAT function 
    cc = lib.stack([
        gcc_values[idxs_grid + n_bin]
        for n_bin in range(-n_correlation_neighbours//2, n_correlation_neighbours//2) 
    ])
    
    # 3. Select the maximum value.
    srp_grid = max_func(cc)

    return srp_grid


def _n_corr_neighbours_srp_batch(
    gcc_idxs,
    gcc_values,
    theoretical_tdoa_grids,
    n_correlation_neighbours,
    lib,
):
    """Vectorized max-neighbour SRP for a batch of pairwise GCC vectors."""
    idxs_grid = lib.searchsorted(gcc_idxs, theoretical_tdoa_grids)
    if lib is torch:
        offsets = torch.arange(
            -n_correlation_neighbours // 2,
            n_correlation_neighbours // 2,
            device=gcc_values.device,
        )
    else:
        offsets = np.arange(
            -n_correlation_neighbours // 2,
            n_correlation_neighbours // 2,
        )
    if lib is torch:
        gather_indices = idxs_grid.unsqueeze(1) + offsets.reshape(
            (1, -1) + (1,) * (idxs_grid.ndim - 1)
        )
    else:
        gather_indices = np.expand_dims(idxs_grid, 1) + offsets.reshape(
            (1, -1) + (1,) * (idxs_grid.ndim - 1)
        )
    if lib is torch:
        gather_indices = gather_indices.clamp(0, gcc_values.shape[1] - 1)
    else:
        gather_indices = np.clip(gather_indices, 0, gcc_values.shape[1] - 1)

    if lib is torch:
        values = gcc_values.gather(
            1, gather_indices.reshape(gcc_values.shape[0], -1)
        )
        return values.reshape(gather_indices.shape).amax(dim=1)

    values = np.take_along_axis(
        np.broadcast_to(gcc_values[:, None, :], (
            gcc_values.shape[0],
            gather_indices.shape[1],
            gcc_values.shape[1],
        )),
        gather_indices.reshape(gcc_values.shape[0], gather_indices.shape[1], -1),
        axis=2,
    )
    return values.reshape(gather_indices.shape).max(axis=1)


def _interpolated_srp(gcc_idxs,
                      gcc_values,
                      theoretical_tdoa_grid):

    if isinstance(gcc_idxs, torch.Tensor):
        srp_grid = torch_interp1d(gcc_idxs, gcc_values, theoretical_tdoa_grid)
    elif isinstance(gcc_idxs, np.ndarray):
        interp_cc = np_interp1d(gcc_idxs, gcc_values, kind='cubic')
        srp_grid = interp_cc(theoretical_tdoa_grid)

    return srp_grid
