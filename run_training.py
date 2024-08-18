import transformer_lens
import torch
import numpy as np
import matplotlib.pyplot as plt
from pruned_tqs.Hamiltonian import Ising
from pruned_tqs.optimizer import Optimizer


if torch.cuda.is_available():
    print("using gpu")
    device = torch.device("cuda")
else:
    print("using cpu")
    device = torch.device("cpu")

system_size = 10

cfg = transformer_lens.HookedTransformerConfig(
    d_model=32,
    d_head=8,
    n_layers=2,
    n_ctx=1 + system_size,
    n_heads=4,
    d_mlp=None,
    d_vocab=3,
    act_fn="relu",
    attn_only=True,
    d_vocab_out=1,
    device="cpu",  # "cuda",
)

hooked_model = transformer_lens.HookedTransformer(cfg)

hooked_model.to(device)

system_size = hooked_model.system_size

Hamiltonians = [Ising(system_size, periodic=False)]

Hamiltonians[0].param_range *= 0
Hamiltonians[0].param_range -= 1.0

param_dim = Hamiltonians[0].param_dim

print("this is param range", Hamiltonians[0].param_range)

assert Hamiltonians[0].param_range.max() < 0

optim = Optimizer(hooked_model, Hamiltonians, point_of_interest=None)

optim.train(
    500_000,  # 500_000,
    title="ten_site_test",
    description_string="simple_test_on_small_computer",
    batch=5000,  # 20000,
    max_unique=10,  # 10,
    param_range=None,  # use the hamiltonian's
    fine_tuning=False,
    use_SR=False,
    ensemble_id=int(False),
    check_kl=True,
)
