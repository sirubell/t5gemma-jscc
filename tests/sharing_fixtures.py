import torch
from model_helpers import tiny_backbone
from jscc.models.channel import AWGNChannel
from jscc.models.codec import Codec
from jscc.models.split_model import SplitModel
from jscc.sharing_protocol import batch_view
from jscc.sharing_schedule import SharingPlan


def sharing_model(cell='D-LN'):
    torch.manual_seed(29)
    architecture, norm = {'D-N': ('direct_affine', 'none'), 'D-none': ('direct_affine', 'none'), 'D-LN': ('direct_outer_ln', 'both'), 'R-none': ('residual_mlp', 'none'), 'R-LN': ('residual_mlp', 'both')}[cell]
    codec = Codec(16, dict(architecture=architecture, hidden_dim=16, bottleneck_dim=8, n_res_blocks=2 if cell.startswith('R') else 0, activation='gelu', dropout=0., layernorm=norm, snr_film=False))
    return SplitModel(tiny_backbone(26), codec, AWGNChannel(), {'stack': 'enc', 'where': 'after_final_norm'}, {'normalize_power': True, 'clean_film_snr': 18.})


def sharing_batches():
    return [dict(input_ids=torch.tensor([[2, 3+i, 4], [2, 5+i, 0]]), attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]), labels=torch.tensor([[5, 6, 1], [7, 1, -100]]), row_ids=torch.tensor([2*i, 2*i+1]), source_family_ids=[f'train-{2*i}', f'train-{2*i+1}']) for i in range(4)]


def sharing_plan(batches=None):
    batches = sharing_batches() if batches is None else batches
    return SharingPlan(tuple(batch_view(b) for b in batches), 'shared-tiny-cpu', synthetic=True)
