#  This source file is part of the Avogadro project.
#  This source code is released under the 3-Clause BSD License, (see "LICENSE").

"""ANI-1x energy and gradient calculator using the binary protocol."""

import json
import sys

import torch
import torchani

from ._ani_server import run_ani_server


def run():
    bootstrap = json.loads(sys.stdin.buffer.readline())
    mol_cjson = bootstrap["cjson"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchani.models.ANI1x(periodic_table_index=True).to(device)

    run_ani_server(mol_cjson, model)
