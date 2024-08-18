import numpy as np
import matplotlib.pyplot as plt
import torch
from .Hamiltonian_utils import bin2dec
from .Hamiltonian import Hamiltonian, Ising
import os
import h5py
import glob

""" 
explicitly compute the values of some states by hand 
compare these to the TQS value
"""


def create_reference_folder(title):
    # Create the directory name by combining a prefix with the title
    directory_path = f"reference_files/reference_folder_{title}"

    # Use os.path.join to ensure the directory path is correctly formatted for the operating system
    # directory_path = os.path.join(os.getcwd(), directory_name)

    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # If the directory does not exist, create it
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
        return True
    else:
        # If the directory exists, inform the user
        print(f"Directory already exists: {directory_path}")
        return False


def compare_to_reference_values(
    states: torch.Tensor, log_probs: torch.tensor, title: str, param: float
):
    batch_size, system_size = states.shape

    idxs, list_log_probs = zip(
        *sorted(
            zip(
                [int(round(x)) for x in bin2dec(states, bits=system_size).tolist()],
                log_probs.tolist(),
            )
        )
    )

    log_probs = torch.tensor(list_log_probs)

    assert system_size <= 15, "system size too large"

    param = str(round(param, 3))

    directory_name = (
        f"reference_files/reference_folder_{title}/param_setting_{param}.h5"
    )

    # Use os.path.join to ensure the directory path is correctly formatted for the operating system
    directory_path = os.path.join(os.getcwd(), directory_name)

    with h5py.File(directory_path, "r") as f:
        dset = f["state"]
        # Suppose we want to access just one slice
        state_vals = np.array([dset[i] for i in idxs])

    # now we'll evaluate the kl divergence
    expected_kl = np.mean(-(log_probs.numpy() - 2 * np.log(state_vals)))
    print(f"param {param} expected kl {expected_kl}")
    return expected_kl


def compute_ising_reference_values(h_values, hamiltonian, title):
    # make reference values
    files_already_there = glob.glob(
        f"reference_files/reference_folder_{title}/param_setting_*.h5"
    )

    for param in h_values:
        param_string = str(round(param, 3))
        directory_name = (
            f"reference_files/reference_folder_{title}/param_setting_{param_string}.h5"
        )

        ws, vs = np.linalg.eigh(hamiltonian.full_H(param=param).todense())
        gs = np.abs(vs[:, 0])

        # Use os.path.join to ensure the directory path is correctly formatted for the operating system
        directory_path = directory_name  # os.path.join(os.getcwd(), directory_name)

        if directory_path not in files_already_there:

            with h5py.File(directory_path, "w") as f:
                dset = f.create_dataset("state", data=gs)
