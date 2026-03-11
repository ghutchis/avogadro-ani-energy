#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""Shared EnergyServer loop for ANI calculators (torchani)."""

import sys

import numpy as np
import torch

from .energy import EnergyServer

# Conversion factors
_HARTREE_TO_KJ_MOL = 2625.4996394799


def run_ani_server(mol_cjson: dict, model) -> None:
    """Run the binary-protocol EnergyServer loop for any torchani model.

    Args:
        mol_cjson: the "cjson" sub-object from the Avogadro bootstrap JSON.
        model: a torchani model already on the target device.
    """
    device = next(model.parameters()).device

    atoms = np.array(mol_cjson["atoms"]["elements"]["number"])
    species = torch.tensor([atoms.tolist()], device=device)
    num_atoms = len(atoms)

    with EnergyServer(sys.stdin.buffer, sys.stdout.buffer, num_atoms) as server:
        for request in server.requests():
            coords = request.coords  # (N, 3) or (batch, N, 3) in Angstrom

            if request.is_batch:
                batch_size = request.batch_size
                coords_tensor = torch.tensor(
                    coords, dtype=torch.float32, device=device
                )
                species_batch = species.expand(batch_size, -1)

                if request.wants_gradient:
                    coords_tensor.requires_grad_(True)
                    energies = model((species_batch, coords_tensor)).energies
                    grads = torch.autograd.grad(
                        energies.sum(), coords_tensor
                    )[0]
                    grads_np = grads.cpu().numpy() * _HARTREE_TO_KJ_MOL
                    request.send_gradients(grads_np)
                else:
                    with torch.inference_mode():
                        energies = model((species_batch, coords_tensor)).energies
                    energies_np = energies.cpu().numpy() * _HARTREE_TO_KJ_MOL
                    request.send_energies(energies_np)
            else:
                coords_tensor = torch.tensor(
                    [coords], dtype=torch.float32, device=device
                )
                need_grad = (
                    request.wants_gradient
                    or request.wants_energy_and_gradient
                )

                if need_grad:
                    coords_tensor.requires_grad_(True)
                    energy = model((species, coords_tensor)).energies
                    grad = torch.autograd.grad(
                        energy.sum(), coords_tensor
                    )[0]
                    grad_np = grad[0].cpu().numpy() * _HARTREE_TO_KJ_MOL
                    energy_kj = energy.item() * _HARTREE_TO_KJ_MOL

                    if request.wants_energy_and_gradient:
                        request.send_energy_and_gradient(energy_kj, grad_np)
                    else:
                        request.send_gradient(grad_np)
                else:
                    with torch.inference_mode():
                        energy = model((species, coords_tensor)).energies
                    request.send_energy(energy.item() * _HARTREE_TO_KJ_MOL)
