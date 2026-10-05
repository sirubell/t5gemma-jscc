"""Scientific fixtures only; real tiny task bodies, routing and channel draws."""

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from jscc import panel_evaluation as panels, panel_workers as core, site_routing
from jscc.experiment_state import CheckpointRef, capture_rng
from jscc.presentation import tensor_digest
from jscc.qualification import state_exact
from jscc.shared_state import recipe_runtime_state





def tiny_runtime(task, checkpoint):
    patches = pytest.MonkeyPatch()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    if task == 'coco':
        from shared.test_shared_coco_scoring import scoring_inputs
        model, processor, data, _, _, request = getattr(scoring_inputs, '__wrapped__')(patches)
    else:
        from datasets import Dataset, DatasetDict
        from contracts.test_harness_payload import fixture_model
        processor, model = fixture_model()
        model.enable_universal_codec()
        def rows(count):
            return [{'ctx_a': 'x', 'ctx_b': 'x', 'activity_label': 'x',
                     'endings': ['x', 'x x', 'x x x', 'x x x x'], 'label': str(i % 4),
                     'source_id': f'original~{i}', 'ind': i} for i in range(count)]
        raw = DatasetDict({'train': Dataset.from_list(rows(8)), 'validation': Dataset.from_list(rows(2))})
        patches.setattr('datasets.load_dataset', lambda *args, **kwargs: raw)
        data = None
        request = {'task': 'hellaswag', 'precision': 'fp32',
                   'backend': model.base.config.decoder._attn_implementation,
                   'input_ids': [0, 1], 'source_family_ids': ['original~0', 'original~1'],
                   'expected_items': 2, 'data_settings': {'name': 'fixture', 'revision': 'fixed'},
                   'noise': {'seed': 13, 'namespace': 'tiny-paired-hellaswag'},
                   'settings': {'batch_size': 8, 'num_fewshot': 5, 'num_samples': 2,
                                'scoring_policy': 'fp32-v1', 'numerical_policy': 'native'}}
    request.update(sites=['enc_emb', 'dec_l0'], conditions=[0.0], expected_items=2)
    model.train()
    for parameter in model.codec.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    reference = core.file_reference(checkpoint)
    state = SimpleNamespace(reference=CheckpointRef(reference['path'], reference['sha256'], reference['bytes']),
        payload={'kind': 'recipe_v1', 'model': deepcopy(site_routing.export_codec(model)),
                 'metadata': {'plan_identity': 'tiny-plan', 'model_revision': 'tiny-scoring',
                    'recipe_plan': {'description': {'config': {'task': task}, 'locations': request['sites']}}}})
    # Fixture-only checkpoint admission/opening. The actual restoration and scorer run.
    patches.setattr('jscc.experiment_state.open_state', lambda *args, **kwargs: state)
    patches.setattr(panels, '_validate_recipe_request', lambda *args, **kwargs:
                    (state, SimpleNamespace(identity='tiny-plan')))
    epsilons = []
    original_randn = torch.randn
    def capture_epsilon(*args, **kwargs):
        epsilon = original_randn(*args, **kwargs)
        if kwargs.get('generator') is not None:
            epsilons.append({'stream': model.channel._stream, 'shape': list(epsilon.shape),
                             'dtype': str(epsilon.dtype), 'epsilon_sha256': tensor_digest(epsilon)})
        return epsilon
    patches.setattr(torch, 'randn', capture_epsilon)
    return {'model': model, 'processor': processor, 'data': data, 'state': state,
            'request': request, 'guard': lambda: None, 'patches': patches,
            'epsilons': epsilons, 'initialize_calls': 0}


@dataclass
class TinyFactory:
    task: str
    checkpoint: str

    def __call__(self, plan, device):
        from jscc import shared_evaluation, shared_preparation, shared_runs, shared_inputs
        runtime = tiny_runtime(self.task, self.checkpoint)
        patches = runtime['patches']
        checkpoint = {k: v for k, v in plan['checkpoint'].items() if k != 'bytes'}
        patches.setattr(shared_evaluation, 'recipe_evaluation_target', lambda *args, **kwargs:
            ({'checkpoint': checkpoint}, runtime['request'], runtime['state'],
             {'model': {'device': 'cpu'}}, 'fixture-initial', {'synthetic_cpu': True}))
        patches.setattr(shared_runs, '_guard', lambda *args, **kwargs: lambda: None)
        def initialize(*args, **kwargs):
            runtime['initialize_calls'] += 1
            return runtime['processor'], runtime['model']
        patches.setattr(shared_preparation, 'initialize_recipe', initialize)
        patches.setattr(shared_inputs, 'load_coco_panel_data', lambda *args: runtime['data'])
        loaded = panels.RecipeRuntimeFactory('fixture-root', plan['case_id'], None, None, 'initial',
            runtime['request'], 60)(plan, device)
        runtime.update(loaded)
        return runtime


def snapshot(runtime):
    model = runtime['model']
    return (deepcopy(model.state_dict()), [None if p.grad is None else p.grad.clone()
            for p in model.codec.parameters()], deepcopy(recipe_runtime_state(model)), capture_rng())


def score_tiny(runtime, plan, panel, output):
    before = snapshot(runtime)
    runtime['epsilons'].clear()
    try:
        result = panels.score_recipe_panel(runtime, plan, panel, output)
    finally:
        restored = state_exact(before, snapshot(runtime))
        Path(output, 'restoration.json').write_text(json.dumps({'restored': restored,
            'pid': os.getpid(), 'initialize_calls': runtime['initialize_calls']})+'\n')
        assert restored
    result.update(actual_epsilons=list(runtime['epsilons']), initialize_calls=runtime['initialize_calls'], pid=os.getpid())
    return result
