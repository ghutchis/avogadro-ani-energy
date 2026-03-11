#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""Entry point for the avogadro-ani-energy plugin.

Avogadro calls this as:
    avogadro-ani-energy <identifier> [--lang <locale>] [--debug]

with the molecule bootstrap JSON on stdin (one compact JSON line).
"""

import argparse


def setup():
    """Download ANI 1x and 2x model parameters by running a test calculation."""
    import numpy as np
    import torch
    import torchani

    device = torch.device('cpu')
    # Water molecule: O, H, H
    species = torch.tensor([[8, 1, 1]], device=device)
    coordinates = torch.tensor([[
        [0.000,  0.000,  0.119],
        [0.000,  0.757, -0.477],
        [0.000, -0.757, -0.477],
    ]], dtype=torch.float32, device=device)

    print("Downloading ANI-1x model parameters (this may take a moment)...")
    model = torchani.models.ANI1x(periodic_table_index=True).to(device)

    with torch.no_grad():
        energy = model((species, coordinates)).energies

    print(f"ANI-1x setup complete. Water energy: {energy.item():.6f} Hartree")

    print("Downloading ANI-2x model parameters (this may take a moment)...")
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    with torch.no_grad():
        energy = model((species, coordinates)).energies

    print(f"ANI-2x setup complete. Water energy: {energy.item():.6f} Hartree")


def main():
    parser = argparse.ArgumentParser("avogadro-ani-energy")
    parser.add_argument("feature")
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--protocol", nargs="?", default="binary-v1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    match args.feature:
        case "ANI1x":
            from .ani1x import run
            run()
        case "ANI2x":
            from .ani2x import run
            run()
