"""Count the explicitly injected tiny dataset backend in each spawned process."""

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import datasets  # noqa: E402
from shared import tiny_parallel_helpers as tiny  # noqa: E402


# Spawned children must select the same two ignored draft modules.



@dataclass
class ResidentFactory:
    checkpoint: str

    def __call__(self, plan, device):
        original_tiny_runtime = tiny.tiny_runtime
        def create(*args):
            runtime = original_tiny_runtime(*args)
            runtime['request']['conditions'] = list(dict.fromkeys(panel['condition'] for panel in plan['panels']))
            runtime['dataset_loads'] = []
            backend = datasets.load_dataset
            def load(*args, **kwargs):
                runtime['dataset_loads'].append((args, kwargs))
                return backend(*args, **kwargs)
            runtime['patches'].setattr(datasets, 'load_dataset', load)
            return runtime
        tiny.tiny_runtime = create
        try:
            return tiny.TinyFactory('hellaswag', self.checkpoint)(plan, device)
        finally:
            tiny.tiny_runtime = original_tiny_runtime


def resident_state(runtime):
    task = runtime['data']['task']
    fewshot_rng = getattr(task, 'fewshot_rnd', None)
    return ('_instances' in task.__dict__, copy.deepcopy(getattr(task, '_instances', None)),
            task.sampler.rnd.getstate(), None if fewshot_rng is None else fewshot_rng.getstate(),
            id(task.dataset))


def score_resident(runtime, plan, panel, output):
    before = resident_state(runtime)
    try:
        result = tiny.score_tiny(runtime, plan, panel, output)
    finally:
        assert resident_state(runtime) == before
    output = Path(output)
    result['task_evidence'] = {name: [json.loads(line) for line in (output/name).read_text().splitlines()]
        for name in ('prompts.jsonl', 'fewshots.jsonl')}
    receipt = json.loads(Path(result['scientific_receipt']['path']).read_text())
    result['payload_counts'] = receipt['payload']
    result['dataset_loads'] = len(runtime['dataset_loads'])
    return result
