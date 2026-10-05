"""Explicit saved protocol, prepared-package and checkpoint readers."""

from __future__ import annotations



from jscc.qualification import check_reconstruction_recipe





from .activation_replay import canonical_digest
from .shared_inputs import batch_identity
from dataclasses import dataclass
import math
from .experiment_schedule import CONDITIONS, NoiseKey
from .shared_inputs import BatchView
from .shared_plan import ScheduledUpdate
import copy
from .asset_paths import asset_file_path
import hashlib
import json
import os
from pathlib import Path
import torch
import yaml
from .activation_replay import file_digest
from .shared_inputs import read_prepared_batch
from .config import load_config
from .data.hellaswag_prompts import PromptBuilder
from .presentation import PresentationSampler
from .asset_identity import _reference, state_dict_identity
from .asset_identity import bind_model_assets
from .shared_inputs import _native_batch
from .shared_inputs import batch_view
import time
from .asset_identity import saved_source_inventory
import zipfile
from .shared_inputs import _check_batches
from .batch_partition import partition_policy
import io
from .qualification import state_exact
from dataclasses import asdict
from .experiment_state import CheckpointRef, open_state
from .models.split_model import ResolvedSite
import tempfile
from .experiment_records import canonical_bytes, completion_status, read_events, verify_inventory
from .presentation import training_source_digest
from .experiment_records import observation_identity
from datetime import datetime, timezone
import importlib.metadata
import socket
import sys
from .qualification import controller_binding_identity as binding_identity, read_controller_reference as _read_ref
from .config import validate_config










# Transferred from baseline_protocol; serialized meanings remain unchanged.
def measure(value=None, reason=None):
    if value is None and not reason:
        raise ValueError("unavailable measurement requires a reason")
    return {"value": value, "reason": reason}


def objective_settings(kind):
    if kind not in {"combined", "local"}:
        raise ValueError("objective must be combined or local; cached functional suffix is disabled")
    return {"temperature": 1.0, "kl_weight": 1.0 if kind == "combined" else 0.0,
            "mse_weight": 0.1, "loss_weights": {
                "kl": 1.0 if kind == "combined" else 0.0, "hidden": 0.1, "memory": 0.0}}


def comparison_refs(learner, metadata, kind):
    """Derive immutable controls from actual configured codec and bound state."""
    result = {
        "architecture": canonical_digest(learner.model.codec.config),
        "initialization": metadata["initialization_identity"],
        "training_data": canonical_digest({"data": metadata.get("data_identity", learner.identity["data"]),
                                           "stream": metadata["stream_identity"]}),
        "objective": canonical_digest(objective_settings(kind)),
        "exposure": canonical_digest({"effective_batch": learner.effective_batch,
                                      "stream": metadata["stream_identity"], "horizon": 400}),
        "schedule": canonical_digest({"schema": "baseline-lr-v2", "horizon": 400, "warmup": 20,
                                      "pairing": learner.pairing_id}),
    }
    declared = metadata.get("comparison")
    if declared is not None and declared != result:
        raise ValueError("declared comparison controls differ from actual config/state")
    return result


def validate_task_receipt(receipt, expected_request):
    """A complete flag cannot replace exact immutable observation evidence."""
    from .evaluation import codec_observation_identity
    expected_identity = codec_observation_identity(expected_request)
    if receipt.get("request") != expected_request or receipt.get("identity") != expected_identity:
        raise ValueError("task observation identity differs from required panel/noise/scorer contract")
    rows = receipt.get("conditions", [])
    if receipt.get("status") != "complete" or [r.get("condition") for r in rows] != expected_request["conditions"]:
        raise ValueError("task observation conditions incomplete")
    count = expected_request["expected_items"]
    for row in rows:
        if row.get("status") != "complete" or any(row.get(key) != count for key in ("requested", "completed", "denominator")) or row.get("failed") != 0:
            raise ValueError("task observation item accounting incomplete")
        items = row.get("items", [])
        ids = [item.get("sample_id", item.get("image_id")) for item in items]
        families = [item.get("source_id", item.get("image_id")) for item in items]
        if ids != expected_request["input_ids"] or families != expected_request["source_family_ids"]:
            raise ValueError("task observation item/source membership mismatch")
        if not row.get("metrics") or row.get("error"):
            raise ValueError("task observation scores unavailable or condition failed")


def open_baseline_replay(directory, requirement, *, source_identity, role, site, backbone, dtype):
    """Bind a producer bank to this consumer, not to self-asserted requirements."""
    from dataclasses import asdict
    from .activation_replay import open_replay
    capability = {"optimization": "local_reconstruction", "objective_validation": "objective_validation"}.get(role)
    if (capability is None or requirement.get("capability") != capability
            or requirement.get("data_role") != role
            or requirement.get("learner_source_sha256") != source_identity):
        raise ValueError("replay consumer source/role/capability differs from current learner")
    replay = open_replay(directory, requirement)
    producer = replay.manifest["producer"]
    expected_site = {**asdict(site), "site_id": site.site_id, "split": site.split}
    if (producer["sites"] != [expected_site] or producer["backbone"] != backbone
            or producer["environment"]["activation_dtype"] != str(dtype)):
        raise ValueError("replay backbone/site/precision differs from actual consumer binding")
    return replay


def read_replay_batch(replay, reference):
    batch = replay.read(reference["batch_view_id"], "enc_fn")
    if batch_identity(batch) != reference["view_sha256"]:
        raise ValueError("replayed input/view differs from prepared ordered stream")
    return batch


# Transferred from sharing_schedule; serialized meanings remain unchanged.
TRAINED_SITES = ("enc_l9", "enc_l19", "enc_fn")


HELDOUT_SITE = "enc_l14"


DEPTH_PROTOCOL = "sharing-dn-q400-effective64-v1"


NATIVE16_PROTOCOL = "sharing-dn-native16-fresh-v1"


NATIVE16_LONG_PROTOCOL = "sharing-dn-native16-long-v1"


ENCODER_PILOT_PROTOCOL = "encoder-original6-native16-pilot-v1"


NATIVE16_PROTOCOLS = (NATIVE16_PROTOCOL, NATIVE16_LONG_PROTOCOL)


NATIVE16_WEIGHT_DECAY = 0.002500001398541448


def validate_native16_horizon(q):
    if type(q) is not int or not 4 <= q <= 6400 or q % 4:
        raise ValueError("native16 requires explicit q4..6400 in complete quarters")
    return q


def native16_exposure(q):
    validate_native16_horizon(q)
    return {"q_per_site": q, "effective_batch": 16, "presentations_per_site": 16 * q,
            "specialist_updates": q, "shared_updates": 3 * q,
            "schedule": "full-horizon-warmup5pct-cosine-terminal-v1"}


def sharing_lr_factor(index: int, horizon: int) -> float:
    """LR used by update index; horizon itself denotes the terminal next LR."""
    if type(horizon) is not int or horizon < 4:
        raise ValueError("horizon must be at least four")
    if type(index) is not int or not 0 <= index <= horizon:
        raise ValueError("schedule index outside finite horizon")
    warmup = max(1, horizon // 20)
    return index / warmup if index < warmup else 0.5 * (
        1 + math.cos(math.pi * (index - warmup) / (horizon - warmup))
    )


@dataclass(frozen=True)
class Observation:
    purpose: str
    learner: str
    site: str
    step: int
    stage: str
    conditions: tuple = CONDITIONS


@dataclass(frozen=True)
class SharingPlan:
    views: tuple[BatchView, ...]
    study_pairing_id: str
    synthetic: bool = False
    layer_count: int = 26
    protocol_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.views, tuple) or not self.views:
            raise ValueError("a frozen ordered view tuple is required")
        if type(self.synthetic) is not bool or not self.study_pairing_id.strip():
            raise ValueError("explicit synthetic marker and study identity required")
        if type(self.layer_count) is not int or self.layer_count < 20:
            raise ValueError("pinned layer count must contain every declared site")
        if self.q < 4 or self.q % 4:
            raise ValueError("quota must have four distinct integral quarters")
        if len({v.sequences for v in self.views}) != 1:
            raise ValueError("effective batch size must be constant")
        if self.protocol_id in NATIVE16_PROTOCOLS:
            validate_native16_horizon(self.q)
            if self.protocol_id == NATIVE16_LONG_PROTOCOL and not self.synthetic and self.q != 6400:
                raise ValueError("formal native16 long study requires q6400")
            if self.batch_size != 16 or len({v.view_sha256 for v in self.views}) != self.q:
                raise ValueError("native16 requires distinct complete16 views; cycling subset forbidden")
        elif not self.synthetic:
            quota = 400 if self.protocol_id == DEPTH_PROTOCOL else 200
            if self.q != quota or self.batch_size != 64:
                raise ValueError("production requires q200 or explicit q400 depth protocol and effective batch64")
            if quota == 400 and len({view.view_sha256 for view in self.views}) != 400:
                raise ValueError("q400 requires400 distinct ordered views; cycling q200 is forbidden")

    @property
    def q(self):
        return len(self.views)

    @property
    def batch_size(self):
        return self.views[0].sequences

    @property
    def raw_final_diagnostic(self):
        return f"enc_l{self.layer_count - 1}"

    @property
    def learners(self):
        return tuple(f"specialist_{site}" for site in TRAINED_SITES) + ("shared",)

    def learner(self, name: str):
        return LearnerSchedule(name, self)

    @property
    def saved_states(self):
        return (("common_initialization", 0),) + tuple(
            (name, step) for name in self.learners
            for step in self.learner(name).checkpoint_steps
        )

    @property
    def observations(self):
        result = []
        if self.protocol_id == NATIVE16_LONG_PROTOCOL:
            for site in TRAINED_SITES:
                result.append(Observation("objective", "common_initialization", site, 0,
                                          "pre_freeze", ("no_noise",)))
            for name in self.learners:
                sites = TRAINED_SITES if name == "shared" else (name.removeprefix("specialist_"),)
                for site in sites:
                    for step in self.learner(name).checkpoint_steps:
                        result.append(Observation("objective", name, site, step,
                                                  "pre_freeze", ("no_noise",)))
            return tuple(result)
        for purpose in ("task", "objective"):
            for site in (*TRAINED_SITES, HELDOUT_SITE):
                result.append(Observation(purpose, "common_initialization", site, 0,
                                          "post_freeze" if site == HELDOUT_SITE else "pre_freeze"))
            for name in self.learners:
                schedule = self.learner(name)
                steps = schedule.assessment_steps if purpose == "task" else schedule.checkpoint_steps
                sites = (*TRAINED_SITES, HELDOUT_SITE) if name == "shared" else (name.removeprefix("specialist_"),)
                for site in sites:
                    for step in steps:
                        result.append(Observation(purpose, name, site, step,
                                                  "post_freeze" if site == HELDOUT_SITE else "pre_freeze"))
        return tuple(result)

    @property
    def condition_panels(self):
        return {purpose: sum(len(o.conditions) for o in self.observations if o.purpose == purpose)
                for purpose in ("task", "objective")}

    @property
    def additional_vanilla_panels(self):
        return 0 if self.protocol_id == NATIVE16_LONG_PROTOCOL else 1


@dataclass(frozen=True)
class LearnerSchedule:
    name: str
    plan: SharingPlan

    def __post_init__(self):
        if self.name not in self.plan.learners:
            raise ValueError("undeclared learner or heldout/diagnostic training")

    @property
    def q(self):
        return self.plan.q

    @property
    def batch_size(self):
        return self.plan.batch_size

    @property
    def horizon(self):
        return self.q * (3 if self.name == "shared" else 1)

    @property
    def checkpoint_steps(self):
        if self.plan.protocol_id == NATIVE16_LONG_PROTOCOL:
            return (self.q, 2 * self.q, 3 * self.q) if self.name == "shared" else (self.q // 2, self.q)
        return tuple(self.horizon * fraction // 4 for fraction in (1, 2, 3, 4))

    @property
    def assessment_steps(self):
        return (self.horizon // 2, self.horizon)

    def update_at(self, index: int) -> ScheduledUpdate:
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError("update outside finite horizon")
        if self.name == "shared":
            local, position = divmod(index, 3)
            site = TRAINED_SITES[position if self.plan.protocol_id == NATIVE16_LONG_PROTOCOL
                                 else (local % 3 + position) % 3]
        else:
            local, site = index, self.name.removeprefix("specialist_")
        view = self.plan.views[local]
        key = NoiseKey(self.plan.study_pairing_id, "training", site, local,
                       view.view_sha256, view.layout_sha256,
                       condition="uniform", draw_schema="runtime-awgn-replay-v2")
        return ScheduledUpdate(self.name, index, site, local, view, key,
                               sharing_lr_factor(index, self.horizon))


def long_final_panels(q: int) -> tuple[Observation, ...]:
    """Terminal task/objective targets, with all no-noise panels before SNR work.

    Callers bind task and objective obligations separately to these same frozen
    terminal checkpoints. The held-out site appears only after the study freeze.
    """
    validate_native16_horizon(q)
    targets = tuple((f"specialist_{site}", site, q) for site in TRAINED_SITES) + tuple(
        ("shared", site, 3 * q) for site in (*TRAINED_SITES, HELDOUT_SITE)
    )
    return tuple(Observation("final", learner, site, step, "post_freeze", (condition,))
                 for condition in CONDITIONS for learner, site, step in targets)


# Transferred from encoder_stages_preparation; serialized meanings remain unchanged.
PREPARATION_CONDITIONS = ['no_noise', -6, 0, 6, 18]


PREPARATION_WIDTHS = (512, 1152, 2304)


SEED = 20260920


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _ref(path, root, *, batch=None):
    reference = {'path': str(Path(path).relative_to(root)), 'sha256': file_digest(path)}
    if batch is not None:
        reference['view_sha256'] = batch_identity(batch)
    return reference


def _link(root, reference, output):
    source = _reference(root, reference)
    target = output / reference['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
    return copy.deepcopy(reference)


def reuse_prefix(root, references, sampler, output):
    """Verify actual native rows once, then retain the exact existing bytes."""
    if len(references) * 16 > len(sampler):
        raise ValueError('prefix exceeds requested presentation stream')
    retained, lengths = [], []
    for index, reference in enumerate(references):
        # Reading verifies both the tensor-file checksum and complete view identity.
        path = _reference(root, reference)
        batch = read_prepared_batch(reference, root)
        if (len(batch['input_ids']) != 16
                or batch['row_ids'].tolist() != sampler.actual_ids[index * 16:(index + 1) * 16].tolist()):
            raise ValueError('saved prefix differs from native16 deterministic full-pool order')
        target = output / reference['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        retained.append(copy.deepcopy(reference))
        lengths.append(_lengths(index, batch))
    return retained, lengths


def final_task(template, data_ids):
    """Keep the inspected official membership and change only the condition list."""
    task = copy.deepcopy(template)
    membership = data_ids['final_validation']
    if (task['expected_items'] != 10042 or task['input_ids'] != list(range(10042))
            or task['input_ids'] != membership['input_ids']
            or task['source_family_ids'] != membership['source_family_ids']
            or task['reference_corpus']['split'] != 'validation'
            or task['precision'] != 'bf16' or task['backend'] != 'sdpa'
            or task['scorer'] != 'lm-eval-hellaswag-fp32-v1-acc-and-acc_norm'):
        raise ValueError('full inspected official validation identity differs')
    settings = task['settings']
    if (settings['num_samples'] != 10042 or settings['num_fewshot'] != 5
            or settings['batch_size'] != 16 or settings['numerical_policy'] != 'native'
            or settings['scoring_policy'] != 'fp32-v1'
            or settings['evidence_mode'] != 'compact-v1' or settings['vanilla'] is not False):
        raise ValueError('fixed final scoring settings differ')
    task['conditions'] = list(PREPARATION_CONDITIONS)
    settings['snrs'] = list(PREPARATION_CONDITIONS)
    return task


def _lengths(index, batch):
    return {'index': index, 'source_padded': batch['input_ids'].shape[1],
            'target_padded': batch['labels'].shape[1],
            'source_valid_tokens': int(batch['attention_mask'].sum()),
            'target_valid_tokens': int(batch['labels'].ne(-100).sum()),
            'source_valid_max': int(batch['attention_mask'].sum(1).max()),
            'target_valid_max': int(batch['labels'].ne(-100).sum(1).max())}


def prepare_inputs(*, prefix_prepared, prefix_sha256, pool_manifest, pool_sha256,
                   assets, arrow, initializers, output, q=10000):
    """Extend one matched per-site stream; the integrator binds execution source."""
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer

    prefix_path, pool_path, assets, arrow, output = map(Path,
        (prefix_prepared, pool_manifest, assets, arrow, output))
    prefix_path, pool_path, assets, arrow, output = (
        p.resolve() for p in (prefix_path, pool_path, assets, arrow, output))
    if q != 10000 or file_digest(prefix_path) != prefix_sha256 or file_digest(pool_path) != pool_sha256:
        raise ValueError('q10000 and exact approved prefix/pool manifests required')
    prefix, pool = json.loads(prefix_path.read_text()), json.loads(pool_path.read_text())
    ids = prefix['data_ids']
    if (len(prefix['updates']) != 6400 or len(ids['train_rows']) != 39058
            or ids['train_rows'] != pool['data_ids']['train_rows']
            or prefix['model_revision'] != pool['model_revision']
            or set(map(int, initializers)) != set(PREPARATION_WIDTHS)):
        raise ValueError('approved full pool, q6400 prefix and all three width initializers required')
    if arrow.stat().st_size != pool['dataset_file']['bytes'] or file_digest(arrow) != pool['dataset_file']['sha256']:
        raise ValueError('pinned raw Arrow differs')
    for name, record in pool['tokenizer_assets']['files'].items():
        if asset_file_path(assets, name).stat().st_size != record['bytes'] or file_digest(asset_file_path(assets, name)) != record['sha256']:
            raise ValueError('pinned tokenizer/config differs: ' + name)
    sampler = PresentationSampler(ids['train_rows'], q * 16, seed=SEED)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'encoder-stages-inputs-v1', 'q': q, 'batch_size': 16,
                'model_revision': prefix['model_revision'], 'data_ids': copy.deepcopy(ids),
                'presentation_stream': {'policy': 'epoch-permutations-v1', 'seed': SEED,
                    'total_presentations': q * 16, 'start_presentation': 0,
                    'pool_identity': canonical_digest(ids['train_rows'])},
                'prefix_origin': {'path': str(prefix_path), 'sha256': prefix_sha256},
                'pool_origin': {'path': str(pool_path), 'sha256': pool_sha256},
                'dataset_file': copy.deepcopy(pool['dataset_file']),
                'tokenizer_assets': copy.deepcopy(pool['tokenizer_assets']),
                'launch_authorized': False, 'initializations': {}, 'configs': {}}
    manifest['updates'], lengths = reuse_prefix(prefix_path.parent, prefix['updates'], sampler, output)
    manifest['validation'] = [_link(prefix_path.parent, ref, output) for ref in prefix['validation']]
    task = final_task(json.loads(_reference(prefix_path.parent, prefix['task_request']).read_text()), ids)
    _json(output / 'task.json', task)
    manifest['task_request'] = _ref(output / 'task.json', output)
    base_config = load_config(_reference(prefix_path.parent, prefix['config']))
    for width in PREPARATION_WIDTHS:
        initializer_path, initializer_sha = initializers[str(width)]
        initializer_path = Path(initializer_path).resolve()
        if file_digest(initializer_path) != initializer_sha:
            raise ValueError('initializer package checksum differs')
        initializer = json.loads(initializer_path.read_text())
        if initializer.get('initialization_policy') != {
                'seed': 0, 'fresh_cpu_codec': True, 'baseline_state_reused': False}:
            raise ValueError('fresh seed0 initializer required, never a trained checkpoint')
        initializer_config = load_config(_reference(initializer_path.parent, initializer['config']))
        if initializer_config['codec']['bottleneck_dim'] != width:
            raise ValueError('initializer width differs')
        original_ref = initializer['initialization']
        state_path = _reference(initializer_path.parent, original_ref)
        if state_dict_identity(torch.load(state_path, map_location='cpu', weights_only=True)) != original_ref['state_identity']:
            raise ValueError('initializer tensor identity differs')
        initial_path = output / f'initial-b{width}.pt'
        os.link(state_path, initial_path)
        manifest['initializations'][str(width)] = {**_ref(initial_path, output),
            'state_identity': original_ref['state_identity'], 'seed': 0,
            'paired_stages': ['A', 'E', 'R'],
            'origin': {'path': str(initializer_path), 'sha256': initializer_sha}}
        config = copy.deepcopy(base_config)
        config['codec']['bottleneck_dim'] = width
        config['run']['name'] = f'encoder-stages-b{width}'
        config['training'].update(max_steps=6 * q, schedule_steps=6 * q,
                                  eval_every=6 * q, save_steps=[6 * q])
        config['training']['presentation_stream']['total_presentations'] = q * 16
        config['evaluation']['snrs'] = list(PREPARATION_CONDITIONS)
        config_path = output / f'config-b{width}.yaml'
        config_path.write_text(yaml.safe_dump(config))
        manifest['configs'][str(width)] = {**_ref(config_path, output),
            'config_identity': canonical_digest(load_config(config_path))}
    raw = Dataset.from_file(str(arrow))
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('actual fast-tokenizer offsets required')
    config = AutoConfig.from_pretrained(assets, local_files_only=True)
    first = read_prepared_batch(prefix['updates'][0], prefix_path.parent)
    builder = PromptBuilder(raw, ids, first['prompt_policy'])
    for index in range(6400, q):
        rows = sampler.actual_ids[index * 16:(index + 1) * 16].tolist()
        batch = _native_batch(rows, index, raw, builder, tokenizer, config)
        path = output / f'native16/update-{index:05d}.pt'
        torch.save(batch, path)
        manifest['updates'].append(_ref(path, output, batch=batch))
        lengths.append(_lengths(index, batch))
        if (index + 1) % 100 == 0:
            print(f'Prepared {index + 1}/{q} exact full-pool native16 views', flush=True)
    sampler.save(output, 16)
    _json(output / 'lengths.json', lengths)
    manifest['stream_evidence'] = {'prefix_views_reused_exactly': 6400, 'new_suffix_views': q - 6400,
        'unique_pool_rows': len(set(sampler.actual_ids.tolist())),
        'actual_ids_sha256_le_int64': hashlib.sha256(sampler.actual_ids.numpy().astype('<i8').tobytes()).hexdigest(),
        'source_valid_tokens_per_site': sum(x['source_valid_tokens'] for x in lengths),
        'target_valid_tokens_per_site': sum(x['target_valid_tokens'] for x in lengths),
        'source_valid_max': max(x['source_valid_max'] for x in lengths),
        'target_valid_max': max(x['target_valid_max'] for x in lengths)}
    manifest['checks'] = {'models_constructed': 0, 'activation_cache': False,
                          'CUDA_initialized': torch.cuda.is_initialized(),
                          'final_split': 'already-inspected official validation; no untouched test claim'}
    manifest['manifest_identity'] = canonical_digest(manifest)
    _json(output / 'inputs.json', manifest)
    return output / 'inputs.json'


def resolve_original13_plans(views, *, inputs, input_identity, synthetic=False):
    """Bind all nine plans to actual input/panel metadata, without model loading.

    Source/input-file hashes are checked by the caller at loading. BatchView
    objects must come from the verified native tensor inputs, never estimates.
    """

    from .activation_replay import canonical_digest
    panel = inputs['data_ids']['development']
    if (panel.get('expected_items') != 256 or panel.get('input_ids') != list(range(256))
            or len(panel.get('source_family_ids', [])) != 256
            or panel.get('split') != 'validation'
            or panel.get('panel') != 'f3e5b8ffbaeaae3a775106937170d8f12cb8670b6ecb01f31efe3c7c33179bae'
            or panel.get('noise') != {'namespace': 'encfn-baseline-20260930-development-v2', 'seed': 0}):
        raise ValueError('original13 requires the exact existing256 development panel')
    if not synthetic:
        pool = inputs['data_ids']['train_rows']
        stream = {'policy': 'epoch-permutations-v1', 'seed': SEED, 'total_presentations': 160000,
                  'start_presentation': 0, 'pool_identity': canonical_digest(pool)}
        if len(pool) != 39058 or len(set(pool)) != 39058 or inputs['presentation_stream'] != stream:
            raise ValueError('original13 full training pool/stream differs')
    identity = canonical_digest(panel)
    if not synthetic and identity != 'd83b0856429b508910fab8a42453490b9d4bce3c52e78e17291b64046116980f':
        raise ValueError('original13 development source-family object differs')
    return tuple(Original13Plan(tuple(views), 'original13-seed0-fullpool-v1', ratio, width,
                                input_identity, identity, synthetic)
                 for width in (2304, 1152, 512) for ratio in ORIGINAL13_RATIOS)


def bind_original13_inputs(inputs_path, *, inputs_sha256, output, model_path=None):
    """Prepare source-bound plan/config identities from retained inputs only."""
    from .activation_replay import canonical_digest


    from .shared_inputs import batch_view
    path = Path(inputs_path).resolve()
    if file_digest(path) != inputs_sha256:
        raise ValueError('original13 exact inputs manifest hash required')
    inputs = json.loads(path.read_text())
    if (inputs.get('manifest_identity') != canonical_digest({k: v for k, v in inputs.items() if k != 'manifest_identity'})
            or inputs.get('synthetic_cpu', False) or inputs.get('q') != 10000):
        raise ValueError('original13 complete real q10000 inputs required')
    pool = inputs['data_ids']['train_rows']
    expected = PresentationSampler(pool, 160000, seed=SEED).actual_ids.tolist()
    views = []
    for i, ref in enumerate(inputs['updates']):
        batch = read_prepared_batch(ref, path.parent)
        if batch['row_ids'].tolist() != expected[i*16:(i+1)*16]:
            raise ValueError('original13 actual native input order differs')
        views.append(batch_view(batch))
    plans = resolve_original13_plans(tuple(views), inputs=inputs,
                                     input_identity=inputs_sha256)
    configurations, initializers = {}, {}
    for width in PREPARATION_WIDTHS:
        config = load_config(_reference(path.parent, inputs['configs'][str(width)]))
        validate_original13_config(config, width)
        if (config['model']['revision'] != 'dd0a2683227859151b1730ca3a63087df5b5f39b'
                or config['model']['dtype'] != 'bfloat16' or config['model']['attn_implementation'] != 'sdpa'):
            raise ValueError('original13 pinned production model recipe differs')
        config = copy.deepcopy(config)
        if model_path is not None:
            config['model']['name'] = str(model_path)
        config['training'].update(max_steps=130000, schedule_steps=130000, eval_every=130000,
                                  save_steps=[])
        config['protocol'] = ORIGINAL13_PROTOCOL
        configurations[str(width)] = config
        ref = inputs['initializations'][str(width)]
        initial = torch.load(_reference(path.parent, ref), map_location='cpu', weights_only=True)
        if ref.get('seed') != 0 or state_dict_identity(initial) != ref['state_identity']:
            raise ValueError('original13 fresh width initializer differs')
        initializers[str(width)] = copy.deepcopy(ref)
    model_names = {c['model']['name'] for c in configurations.values()}
    if len(model_names) != 1:
        raise ValueError('original13 widths require the same backbone/tokenizer snapshot')
    assets = bind_model_assets(next(iter(model_names)), 'dd0a2683227859151b1730ca3a63087df5b5f39b',
                              tokenizer_files=inputs['tokenizer_assets']['files'], production=True)
    for config in configurations.values():
        config['model']['name'] = assets['path']
        config['model']['asset_binding'] = copy.deepcopy(assets)
    template = final_task(json.loads(_reference(path.parent, inputs['task_request']).read_text()), inputs['data_ids'])
    template['conditions'] = ['no_noise', -6]
    template['settings']['snrs'] = ['no_noise', -6]
    panel = copy.deepcopy(inputs['data_ids']['development'])
    observation = {**template, 'schema': 'original13-observation-request-v1', 'panel': panel,
                   'input_ids': panel['input_ids'], 'source_family_ids': panel['source_family_ids'],
                   'expected_items': 256, 'noise': panel['noise'],
                   'sites': ['enc_l9', 'enc_fn', 'dec_l0', 'dec_l24'], 'conditions': ['no_noise', -6],
                   'settings': {**template['settings'], 'num_samples': 256, 'snrs': ['no_noise', -6]}}
    inventory = stages_source_inventory()
    from .runtime import source_state
    manifest = {'schema': 'original13-prepared-v1', 'source_commit': source_state()['revision'],
                'source_inventory': inventory,
                'source_identity': canonical_digest(inventory), 'inputs_path': str(path),
                'inputs_sha256': inputs_sha256, 'data_identity': canonical_digest(inputs['data_ids']),
                'configs': configurations, 'initializations': initializers,
                'plans': [plan.resolved() for plan in plans], 'observation_request': observation,
                'terminal_request': template, 'launch_authorized': False}
    manifest['manifest_identity'] = canonical_digest(manifest)
    output = Path(output)
    if output.exists():
        raise ValueError('fresh original13 prepared destination required')
    from .experiment_records import write_manifest
    write_manifest(output, manifest)
    return manifest


def validate_original13_config(config, width):
    """Check the adopted cohort recipe at preparation and runtime load."""
    train, codec, channel = config['training'], config['codec'], config['channel']
    if (config['task'] != 'hellaswag' or config['seed'] != 0
            or codec['architecture'] != 'direct_affine' or codec['bottleneck_dim'] != width
            or codec['layernorm'] != 'none' or codec['n_res_blocks'] != 0 or codec['snr_film']
            or codec['dropout'] != 0 or train['batch_size'] != 16 or train['gradient_accumulation'] != 1
            or train['lr'] != 2e-4 or train['weight_decay'] != 0.002500001398541448
            or train['temperature'] != 1 or train['grad_clip'] != 1 or not train['valid_only_kl']
            or train['warmup_ratio'] != .05 or train['min_lr_ratio'] != 0
            or train.get('patience') is not None or train.get('loss_weights') is not None
            or train['kl_weight'] != 1 or train['mse_weight'] != .1
            or config['model'].get('numerical_policy', 'native') != 'native'
            or codec.get('memory') is not None
            or channel['type'] != 'awgn' or not channel['normalize_power'] or not channel['train_noise']
            or channel['train_snr_range'] != [-6.0, 18.0]):
        raise ValueError('inherited original13 configuration differs')


# Transferred from original13_inputs; serialized meanings remain unchanged.
def load_original13_prepared(path, *, checksum, width, ratio, load_batches=True, allow_pending_assets=False):
    """Load only bound local inputs/configs; never select a newer run or checkpoint."""

    from .asset_identity import verify_model_assets
    from .config import validate_config
    path = Path(path).resolve()
    if file_digest(path) != checksum:
        raise ValueError('exact original13 prepared SHA required')
    manifest = json.loads(path.read_text())
    inventory = stages_source_inventory()
    if (manifest.get('schema') != 'original13-prepared-v1'
            or manifest.get('manifest_identity') != canonical_digest(
                {k: v for k, v in manifest.items() if k != 'manifest_identity'})
            or manifest['source_inventory'] != inventory
            or manifest['source_identity'] != canonical_digest(inventory)):
        raise ValueError('original13 prepared source/manifest differs')
    plans = tuple(Original13Plan.from_resolved(row) for row in manifest['plans'])
    selected = [p for p in plans if p.width == width and p.ratio == ratio]
    if len(selected) != 1:
        raise ValueError('exact declared original13 width/ratio required')
    plan = selected[0]
    inputs_path = Path(manifest['inputs_path'])
    if file_digest(inputs_path) != manifest['inputs_sha256'] or plan.input_identity != manifest['inputs_sha256']:
        raise ValueError('original13 input manifest differs')
    inputs = json.loads(inputs_path.read_text())
    if (manifest['data_identity'] != canonical_digest(inputs['data_ids'])
            or canonical_digest(manifest['observation_request']['panel']) != plan.observation_identity):
        raise ValueError('original13 prepared membership differs')
    config = copy.deepcopy(manifest['configs'][str(width)])
    validate_config(config)
    validate_original13_config(config, width)
    assets = verify_model_assets(config, require_complete=not allow_pending_assets, verify_bytes=False)
    if not plan.synthetic and (not assets['production']
            or assets['tokenizer_files'] != inputs['tokenizer_assets']['files']):
        raise ValueError('original13 production input/model assets differ')
    if (config['protocol'] != ORIGINAL13_PROTOCOL or config['split'] != PILOT_SITE_SPECS['enc_fn']
            or config['training']['max_steps'] != plan.horizon
            or config['training']['schedule_steps'] != plan.horizon):
        raise ValueError('original13 resolved config/horizon differs')
    if not plan.synthetic:
        if (config['model']['revision'] != 'dd0a2683227859151b1730ca3a63087df5b5f39b'
                or config['model']['dtype'] != 'bfloat16' or config['model']['attn_implementation'] != 'sdpa'
                or len(plans) != 9 or set((p.width, p.ratio) for p in plans) != {
                    (w, r) for w in (512, 1152, 2304) for r in (50, 75, 90)}):
            raise ValueError('original13 production model/matrix differs')
        expected = resolve_original13_plans(plan.views, inputs=inputs, input_identity=plan.input_identity)
        if [p.identity for p in expected] != [p.identity for p in plans]:
            raise ValueError('original13 production plan bindings differ')
    initial_ref = manifest['initializations'][str(width)]
    if initial_ref != inputs['initializations'][str(width)] or initial_ref.get('seed') != 0:
        raise ValueError('original13 width initializer origin differs')
    initial = torch.load(_reference(inputs_path.parent, initial_ref), map_location='cpu', weights_only=True)
    if state_dict_identity(initial) != initial_ref['state_identity']:
        raise ValueError('original13 initializer bytes differ')
    batches = None
    if load_batches:
        batches = [read_prepared_batch(ref, inputs_path.parent) for ref in inputs['updates']]
        if tuple(batch_view(b) for b in batches) != plan.views:
            raise ValueError('original13 ordered actual input views differ')
        if not plan.synthetic:
            rows = [int(i) for b in batches for i in b['row_ids']]
            expected_rows = PresentationSampler(inputs['data_ids']['train_rows'], 160000, seed=20260920).actual_ids.tolist()
            if rows != expected_rows:
                raise ValueError('original13 native row stream differs')
    metadata = {'source_identity': manifest['source_identity'], 'source_commit': manifest['source_commit'],
                'model_assets_identity': canonical_digest(assets),
                'config_identity': canonical_digest(config), 'data_identity': manifest['data_identity'],
                'initialization_identity': initial_ref['state_identity'], 'stream_identity': plan.input_identity,
                'protocol_identity': ORIGINAL13_PROTOCOL, 'prepared_identity': manifest['manifest_identity'],
                'model_revision': config['model']['revision']}
    return manifest, config, plan, batches, initial, metadata, inputs['data_ids']


# Transferred from original13_provenance; serialized meanings remain unchanged.
def current_source_inventory(source_root=None):
    """Bind the actual nonlocal code providers of a newly prepared operation.

    Removed campaign fronts cannot be implicit requirements of new bindings.
    Existing saved inventories continue to be checked against their exact bytes.
    """
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    names = {str(path.relative_to(root)) for directory in ('jscc', 'scripts')
             for path in (root/directory).rglob('*')
             if path.is_file() and path.suffix in ('.py', '.sh')}
    names.update(name for name in ('experiment.py', 'train.py', 'evaluate.py', 'study.py',
                                  'uv.lock', 'pyproject.toml') if (root/name).is_file())
    return {name: file_digest(root/name) for name in sorted(names)}


def stages_source_inventory():
    return current_source_inventory()


def original13_execution_guard(config, *, synthetic, timeout_seconds, device_uuid=None):
    """Bound one direct invocation; GPU assignment is checked before model load."""
    import os
    import math
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError('original13 finite positive execution timeout required')
    device = config['model']['device']
    if device == 'cpu':
        if not synthetic:
            raise ValueError('production original13 requires an assigned CUDA target')
    elif device == 'cuda':
        if not device_uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != device_uuid:
            raise ValueError('original13 exact assigned GPU UUID required')
    else:
        raise ValueError('original13 requires explicit cpu or cuda device')
    started = time.monotonic()
    def guard():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError('original13 finite invocation deadline reached')
    return guard


def verify_original13_execution_source(manifest, approved_source):
    import subprocess
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    root = Path(__file__).resolve().parents[1]
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, env=env, text=True).strip()
    if current != approved_source or manifest['source_commit'] != current:
        raise ValueError('original13 exact approved source commit required')
    names = sorted(manifest['source_inventory'])
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', *names], cwd=root,
                   stdout=subprocess.DEVNULL, check=True, env=env)
    subprocess.run(['git', 'diff', '--exit-code', current, '--', *names], cwd=root,
                   stdout=subprocess.DEVNULL, check=True, env=env)


# Transferred from sharing_preparation; serialized meanings remain unchanged.
SHARING_PREPARED_SCHEMA = "sharing-prepared-v1"


SHARING_PREPARED_CELLS = {"D-N": ("direct_affine", "none"), "D-LN": ("direct_outer_ln", "both"), "R-LN": ("residual_mlp", "both"),
         "T-LN": ("two_linear_gelu", "both")}


def write_prepared(path, manifest):
    """Exclusively publish a complete supplied package manifest with its digest.

    All referenced inputs must already exist; verification precedes publication.
    No missing scientific identity or candidate is inferred by this function.
    """
    path = Path(path).resolve()
    value = dict(manifest)
    value.pop("manifest_identity", None)
    value["manifest_identity"] = canonical_digest(value)
    # Source root is always the code actually running this writer.
    _verify(value, path.parent)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def load_prepared(path, *, source_root=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    result = _verify(manifest, path.parent, source_root=source_root)
    result["prepared_path"] = str(path)
    return result


def _verify(manifest, root, *, source_root=None):
    if not isinstance(manifest, dict) or manifest.get("schema") != SHARING_PREPARED_SCHEMA:
        raise ValueError("sharing-prepared-v1 manifest required")
    original = {key: value for key, value in manifest.items() if key != "manifest_identity"}
    if manifest.get("manifest_identity") != canonical_digest(original):
        raise ValueError("manifest identity mismatch")
    required = {"cell", "synthetic_cpu", "q", "batch_size", "protocol_id", "model_revision",
                "study_pairing_id", "config", "config_identity", "updates", "validation",
                "task_request", "geometry", "source_inventory", "source_identity", "source_archive",
                "initialization", "data_ids", "recipe_identity", "architecture_decision", "hard_cap_seconds"}
    if not required <= manifest.keys():
        raise ValueError("incomplete sharing preparation")
    cell, synthetic = manifest["cell"], manifest["synthetic_cpu"]
    if cell not in SHARING_PREPARED_CELLS or type(synthetic) is not bool:
        raise ValueError("one explicit retained candidate and synthetic marker required")
    q, size = manifest["q"], manifest["batch_size"]
    partition = manifest.get("execution_partition")
    if partition is not None and (not isinstance(partition, dict) or partition != partition_policy(partition.get("microbatch_size"))):
        raise ValueError("explicit supported execution partition required")
    native16 = manifest.get("protocol_id") in NATIVE16_PROTOCOLS
    if native16:
        validate_native16_horizon(q)
        if manifest["protocol_id"] == NATIVE16_LONG_PROTOCOL and not synthetic and q != 6400:
            raise ValueError("formal native16 long study requires q6400")
        if partition is not None or size != 16 or (synthetic and q != 4):
            raise ValueError("native16 requires physical16/accum1; CPU lifecycle q4 only")
        if manifest.get("exposure_binding") != native16_exposure(q):
            raise ValueError("native16 requires explicit horizon/exposure binding")
    expected_size = 16 if native16 else 64 if partition is not None or not synthetic else 2
    quota = 400 if manifest.get("protocol_id") == DEPTH_PROTOCOL else 200
    if type(q) is not int or type(size) is not int or (q, size) != (q if native16 else 4 if synthetic else quota, expected_size):
        raise ValueError("production requires q200 or explicit q400/native64; synthetic CPU requires q4/batch2")
    for key in ("protocol_id", "model_revision", "study_pairing_id", "recipe_identity", "architecture_decision"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"explicit {key} required")
    cap = manifest["hard_cap_seconds"]
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("finite positive hard cap required")
    inventory = saved_source_inventory(source_root) if source_root is not None else current_source_inventory()
    if manifest["source_inventory"] != inventory or manifest["source_identity"] != canonical_digest(inventory):
        raise ValueError("executed source inventory/identity differs from preparation")
    archive = _reference(root, manifest["source_archive"])
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(inventory) or set(bundle.namelist()) != set(inventory):
            raise ValueError("source archive must contain exact declared inventory")
        for name, digest in inventory.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise ValueError("retained source archive bytes differ")
    config_path = _reference(root, manifest["config"])
    # A resolved snapshot cannot hide an unbound model-design include.
    if "model_config" in yaml.safe_load(config_path.read_text()):
        raise ValueError("prepared config must be a complete resolved snapshot")
    config = load_config(config_path)
    if canonical_digest(config) != manifest["config_identity"]:
        raise ValueError("resolved config identity mismatch")
    _config_contract(config, manifest)
    updates = manifest["updates"]
    if not isinstance(updates, list) or len(updates) != q:
        raise ValueError("complete ordered update stream required")
    views = []
    for reference in updates:
        _reference(root, reference)
        view = batch_view(read_prepared_batch(reference, root))
        if view.sequences != size:
            raise ValueError("update differs from declared native batch size")
        views.append(view)
    SharingPlan(tuple(views), manifest["study_pairing_id"], synthetic=synthetic,
                protocol_id=manifest["protocol_id"])
    if native16 and not synthetic:
        read_native16_readiness_cases(root, manifest.get("readiness_cases"))
    validation = _batch_group(root, manifest["validation"])
    if sum(view.sequences for view in validation) != (2 if synthetic else 128):
        raise ValueError("objective validation panel count mismatch")
    geometry = manifest["geometry"]
    if not isinstance(geometry, dict) or set(geometry) != {"train", "selection"}:
        raise ValueError("geometry requires train and selection references; no heldout capture")
    for partition in geometry.values():
        group = _batch_group(root, partition)
        if sum(view.sequences for view in group) != (2 if synthetic else 128):
            raise ValueError("geometry panel count mismatch")
    _geometry_contract(root, geometry)
    task_template = json.loads(_reference(root, manifest["task_request"]).read_text())
    validate_task_template(task_template, config, synthetic=synthetic)
    if manifest.get("protocol_id") == NATIVE16_LONG_PROTOCOL:
        if not synthetic and q != 6400:
            raise ValueError("formal native16 long study requires q6400")
        final_template = json.loads(_reference(root, manifest.get("final_task_request")).read_text())
        validate_task_template(final_template, config, synthetic=synthetic, final=True)
    if not synthetic:
        _production_data_contract(root, manifest, task_template)
    initialization = manifest["initialization"]
    state = torch.load(_reference(root, initialization), map_location="cpu", weights_only=True)
    if state_dict_identity(state) != initialization.get("state_identity"):
        raise ValueError("initial codec state identity mismatch")
    if not synthetic and (native16 or manifest["q"] == 400):
        from .models.codec import Codec
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            expected_initial = Codec(1152, config["codec"]).state_dict()
        if state_dict_identity(state) != state_dict_identity(expected_initial):
            raise ValueError("q400/native16 initialization must be fresh seed0; trained state cannot be reused")
    acceptance = manifest.get("gpu_acceptance")
    if acceptance is not None:
        _acceptance_receipt(root, manifest)
    return {**manifest, "resolved_config": config, "task_template": task_template,
            "initialization_state": state, "prepared_identity": manifest["manifest_identity"]}


def _batch_group(root, references):
    if not isinstance(references, list) or not references:
        raise ValueError("nonempty batch reference panel required")
    result = []
    for reference in references:
        _reference(root, reference)
        result.append(batch_view(read_prepared_batch(reference, root)))
    return result


def read_native16_readiness_cases(root, cases):
    """Check immutable diagnostic inputs on CPU before constructing a model."""
    if (not isinstance(cases, list) or len(cases) != 3
            or any(not isinstance(case, dict) for case in cases)
            or [case.get("label") for case in cases]
            != ["representative", "longest_source", "longest_target"]):
        raise ValueError("explicit native16 readiness cases required")
    batches, identities = [], []
    for case in cases:
        reference = case.get("batch")
        _reference(root, reference)
        batch = read_prepared_batch(reference, root)
        _check_batches([batch], "combined", 16)
        view = batch_view(batch)
        if view.sequences != 16:
            raise ValueError("native16 readiness case requires physical16")
        batches.append(batch)
        identities.append(view.view_sha256)
    if len(set(identities)) != 3:
        raise ValueError("three distinct native16 readiness cases required")
    return batches


def _config_contract(config, manifest):
    synthetic = manifest["synthetic_cpu"]
    codec, model, training = config["codec"], config["model"], config["training"]
    if (codec["architecture"], codec["layernorm"]) != SHARING_PREPARED_CELLS[manifest["cell"]]:
        raise ValueError("prepared candidate/config topology mismatch")
    if (config["task"] != "hellaswag" or config["seed"] != 0
            or config["split"] != {"stack": "enc", "where": "after_final_norm"}
            or model["revision"] != manifest["model_revision"]
            or model.get("numerical_policy", "native") != "native"):
        raise ValueError("config violates sharing task/seed/site/model contract")
    native16 = manifest.get("protocol_id") in NATIVE16_PROTOCOLS
    if native16:
        q = validate_native16_horizon(manifest.get("q"))
        if (manifest.get("cell") != "D-N" or manifest.get("batch_size") != 16
                or manifest.get("execution_partition") is not None
                or manifest.get("exposure_binding") != native16_exposure(q)
                or manifest.get("recipe_identity") != "K+0.1R-native-effective16-fresh-v1"
                or manifest.get("initialization_policy") != {
                    "seed": 0, "fresh_cpu_codec": True, "baseline_state_reused": False}
                or config.get("protocol") != manifest["protocol_id"]
                or training.get("max_steps") != q or training.get("schedule_steps") != q
                or training.get("warmup_ratio") != .05 or training.get("min_lr_ratio") != 0
                or training.get("valid_only_kl") is not True or training.get("loss_weights") is not None
                or training.get("patience") is not None):
            raise ValueError("native16 requires explicit fresh D-N terminal horizon/exposure recipe")
        if manifest["protocol_id"] == NATIVE16_LONG_PROTOCOL and not synthetic:
            if (model["device"] != "cuda" or model["dtype"] != "bfloat16"
                    or model.get("attn_implementation") != "sdpa"
                    or config["channel"].get("train_noise") is not True
                    or config["channel"].get("train_snr_range") != [-6, 18]):
                raise ValueError("formal long study requires native BF16/SDPA and uniform -6..18 dB training")
    elif manifest.get("q") == 400 or manifest.get("protocol_id") == DEPTH_PROTOCOL:
        if (synthetic or manifest.get("q") != 400 or manifest.get("protocol_id") != DEPTH_PROTOCOL
                or manifest.get("cell") != "D-N" or manifest.get("selected_bottleneck_dim") != 512
                or manifest.get("batch_size") != 64 or codec["bottleneck_dim"] != 512
                or manifest.get("execution_partition") != partition_policy(16)
                or manifest.get("recipe_identity") != "K+0.1R-native-effective64-q400"
                or manifest.get("initialization_policy") != {
                    "seed": 0, "fresh_cpu_codec": True, "baseline_state_reused": False}):
            raise ValueError("q400 is only the explicit fresh seed0 D-N/B512 effective64/16x4 depth pilot")
    if synthetic and (model["device"] != "cpu" or model["dtype"] != "float32"):
        raise ValueError("synthetic fixture must use CPU/float32")
    partition = manifest.get("execution_partition")
    width = codec["bottleneck_dim"]
    if type(width) is not int or width <= 0:
        raise ValueError("positive integer bottleneck required")
    if not synthetic:
        if (codec.get("hidden_dim") != 1152 or codec.get("activation") != "gelu"
                or codec.get("n_res_blocks") != (2 if manifest["cell"] == "R-LN" else 0)):
            raise ValueError("production codec differs from reviewed D-N/D-LN/R-LN/T-LN topology")
        if manifest.get("selected_bottleneck_dim") != width or width not in (512, 1152, 2304):
            raise ValueError("production requires explicit reviewed bottleneck binding")
        if partition is None and not native16:
            raise ValueError("production requires explicit qualified execution partition")
    if partition is None:
        microbatch, accumulation = manifest["batch_size"], 1
    else:
        if partition != partition_policy(partition.get("microbatch_size")) or manifest["batch_size"] != 64:
            raise ValueError("partition requires parent effective64")
        if manifest.get("selected_bottleneck_dim") != codec["bottleneck_dim"]:
            raise ValueError("explicit selected bottleneck binding required")
        microbatch, accumulation = partition["microbatch_size"], partition["gradient_accumulation"]
    if (training["batch_size"] != microbatch or training["gradient_accumulation"] != accumulation
            or training["lr"] != 2e-4 or training["weight_decay"] != (NATIVE16_WEIGHT_DECAY if native16 else .01)
            or training["grad_clip"] != 1 or training["temperature"] != 1
            or training["kl_weight"] != 1 or training["mse_weight"] != .1):
        raise ValueError("config differs from accepted native-batch optimizer/objective")
    if (codec["snr_film"] or codec["dropout"] != 0 or config["channel"]["type"] != "awgn"
            or not config["channel"]["normalize_power"]):
        raise ValueError("sharing requires dropout0/FiLMoff/masked normalized AWGN")


def _acceptance_receipt(root, manifest):
    receipt = json.loads(_reference(root, manifest["gpu_acceptance"]).read_text())
    if not isinstance(receipt, dict) or receipt.get("status") != "accepted" or any(
        receipt.get(key) != manifest[key]
        for key in ("source_identity", "config_identity", "recipe_identity", "model_revision")
    ):
        raise ValueError("runtime acceptance is not bound to this preparation")
    return receipt




def validate_task_template(template, config, *, synthetic, final=False, no_noise_only=False):
    required = {"task", "expected_items", "conditions", "input_ids", "source_family_ids",
                "prompt_policy", "noise", "layout", "backend", "precision", "scorer",
                "reference_corpus", "settings", "data_settings"}
    if not isinstance(template, dict) or not required <= template.keys():
        raise ValueError("task template requires complete static observation fields")
    count = template["expected_items"]
    if type(count) is not int or count != (2 if synthetic else 10042 if final else 256) or template["task"] != "hellaswag":
        raise ValueError("task request must bind the fixed HellaSwag panel")
    if final and not synthetic and template["input_ids"] != list(range(10042)):
        raise ValueError("final evaluation requires the full ordered official validation membership")
    if template["conditions"] != (["no_noise"] if no_noise_only else ["no_noise", -6, 0, 6, 12, 18]):
        raise ValueError("task request conditions differ from the fixed evaluation scope")
    ids, families = template["input_ids"], template["source_family_ids"]
    if (not isinstance(ids, list) or not isinstance(families, list)
            or len(ids) != count or len(families) != count
            or any(type(value) not in (str, int) or value == "" for value in [*ids, *families])
            or len(set(ids)) != count):
        raise ValueError("task requires unique input IDs and actual source-family IDs")
    for field in ("prompt_policy", "layout", "backend", "precision", "scorer", "reference_corpus"):
        if not template[field]:
            raise ValueError(f"task template identity requires {field}")
    noise = template["noise"]
    if (not isinstance(noise, dict) or type(noise.get("seed")) is not int
            or not isinstance(noise.get("namespace"), str) or not noise["namespace"].strip()):
        raise ValueError("task requires explicit noise seed and namespace")
    settings = template["settings"]
    if (not isinstance(settings, dict) or type(settings.get("num_samples")) is not int
            or settings["num_samples"] != count or settings.get("numerical_policy", "native") != "native"
            or not isinstance(template["data_settings"], dict)):
        raise ValueError("task settings require exact sample count and native numerical policy")
    if final and not synthetic and (settings.get("batch_size") != 16
                                   or settings.get("num_fewshot") != 5
                                   or settings.get("scoring_policy") != "fp32-v1"):
        raise ValueError("final full validation requires batch16/five-shot/FP32 scoring")
    dtype = config["model"]["dtype"]
    alias = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}.get(dtype)
    if template["precision"] not in (alias, f"torch.{dtype}"):
        raise ValueError("task precision differs from configured backbone")
    backend = config["model"].get("attn_implementation")
    if backend is not None and template["backend"] != backend:
        raise ValueError("task backend differs from configured backbone")
    if settings.get("attention_backend", template["backend"]) != template["backend"]:
        raise ValueError("task settings and declared backend differ")


def _geometry_contract(root, geometry):
    from .geometry_capture import _batch_metadata

    queries = {}
    for partition, references in geometry.items():
        seen = set()
        queries[partition] = set()
        for reference in references:
            batch = read_prepared_batch(reference, root)
            required = {"query_ids", "view_ids", "token_positions", "token_roles"}
            if not required <= batch.keys():
                raise ValueError("geometry batch requires query/view/token positions and roles")
            _batch_metadata(batch)
            pairs = set(zip(batch["query_ids"], batch["view_ids"]))
            if seen & pairs:
                raise ValueError("duplicate geometry query/view across batches")
            seen |= pairs
            queries[partition].update(batch["query_ids"])
    if queries["train"] & queries["selection"]:
        raise ValueError("geometry training and selection queries must be disjoint")


def preview(manifest):
    if manifest["protocol_id"] == NATIVE16_LONG_PROTOCOL:
        return {"schema": SHARING_PREPARED_SCHEMA, "cell": manifest["cell"], "synthetic_cpu": manifest["synthetic_cpu"],
                "physical_updates": 6 * manifest["q"], "learned_saves": 9, "initializations": 1,
                "task_panels": 0, "vanilla_panels": 0, "objective_panels": 18,
                "final_full_validation_panels": 42, "final_vanilla_panels": 1, "final_order": "all no_noise then -6/0/6/12/18",
                "gpu_execution_ready": False, "prepared_identity": manifest["prepared_identity"]}
    return {"schema": SHARING_PREPARED_SCHEMA, "cell": manifest["cell"], "synthetic_cpu": manifest["synthetic_cpu"],
            "physical_updates": manifest["q"] * 6, "learned_saves": 16, "initializations": 1,
            "task_panels": 108, "vanilla_panels": 1, "objective_panels": 192,
            "gpu_execution_ready": False,
            "prepared_identity": manifest["prepared_identity"]}


def _production_data_contract(root, manifest, task_template):
    """Bind production roles to retained v2 producers and actual native views.

    Ordered manifest references may repeat a producer view. The producer's
    unique view inventory follows their first-occurrence order. Dataset row IDs
    are scoped by their existing train/validation/development partitions.
    """
    from .activation_replay import _validate_producer_v2, sequence_view_v2

    role_references = {
        "optimization": manifest["updates"],
        "objective_validation": manifest["validation"],
        "geometry_training": manifest["geometry"]["train"],
        "geometry_selection": manifest["geometry"]["selection"],
    }
    producers = manifest.get("producers")
    if not isinstance(producers, dict) or set(producers) != set(role_references):
        raise ValueError("production requires all four role-bound v2 producer references")
    ids = manifest["data_ids"]
    required_ids = ("train_rows", "validation_rows", "objective_rows")
    if not isinstance(ids, dict) or any(key not in ids for key in required_ids):
        raise ValueError("production data_ids require explicit train/validation/objective/development rows")
    for key in required_ids:
        rows = ids[key]
        if (not isinstance(rows, list) or not rows or any(type(row) is not int or row < 0 for row in rows)
                or len(set(rows)) != len(rows)):
            raise ValueError("data role memberships must contain unique nonnegative row IDs")
    catalogs = {}
    for partition in ("train", "validation"):
        families = ids.get(f"{partition}_source_family_ids")
        rows = ids[f"{partition}_rows"]
        if (not isinstance(families, list) or len(families) != len(rows)
                or any(not isinstance(family, str) or not family for family in families)):
            raise ValueError("complete row-aligned training/selection source-family catalogs required")
        catalogs[partition] = dict(zip(rows, families))
    if set(ids["train_rows"]) & set(ids["validation_rows"]):
        raise ValueError("optimization and selection row pools overlap")
    if not set(ids["objective_rows"]) <= set(ids["validation_rows"]):
        raise ValueError("objective rows must belong to the selection pool")
    development = ids.get("development")
    if (not isinstance(development, dict)
            or development.get("input_ids") != task_template["input_ids"]
            or development.get("source_family_ids") != task_template["source_family_ids"]):
        raise ValueError("development membership differs from actual task panel")
    role_rows, role_families, role_demos, specs = {}, {}, {}, {}
    for role, references in role_references.items():
        spec = json.loads(_reference(root, producers[role]).read_text())
        _validate_producer_v2(spec)
        if (spec["data_role"] != role or spec["data"]["task"] != "hellaswag"
                or spec["backbone"]["model_revision"] != manifest["model_revision"]):
            raise ValueError("producer task/model/data role differs from preparation")
        by_view = {view["batch_view_id"]: view for view in spec["views"]}
        ordered_views, rows, families, demos = [], [], set(), set()
        for reference in references:
            _reference(root, reference)
            batch = read_prepared_batch(reference, root)
            try:
                view = sequence_view_v2(batch)
            except (KeyError, AttributeError, TypeError) as error:
                raise ValueError("production batch lacks complete native v2 view provenance") from error
            view_id = view["batch_view_id"]
            if view != by_view.get(view_id):
                raise ValueError("actual ordered batch differs from role-bound producer view")
            if view_id not in ordered_views:
                ordered_views.append(view_id)
            batch_rows = batch["row_ids"].tolist()
            count = len(batch["input_ids"])
            if (not isinstance(batch_rows, list) or len(batch_rows) != count
                    or any(type(row) is not int or row < 0 for row in batch_rows)
                    or any(len(view[key]) != count for key in
                           ("view_ids", "source_family_ids", "demo_ids", "source_views"))):
                raise ValueError("producer provenance must bind every native sequence")
            if any(not isinstance(family, str) or not family for family in view["source_family_ids"]):
                raise ValueError("actual source families are required")
            catalog = catalogs["train" if role in ("optimization", "geometry_training") else "validation"]
            if any(catalog.get(row) != family for row, family in zip(batch_rows, view["source_family_ids"])):
                raise ValueError("actual query row/source family differs from complete role catalog")
            for row, source_view, demo_ids in zip(batch_rows, view["source_views"], view["demo_ids"]):
                demo_families = source_view.get("demo_source_family_ids")
                if (source_view.get("row_id") != row or source_view.get("native_prefix") != "reference"
                        or not isinstance(demo_families, list) or not isinstance(demo_ids, list)
                        or len(demo_families) != len(demo_ids)
                        or any(not isinstance(family, str) or not family for family in demo_families)):
                    raise ValueError("native source-view row/demo provenance mismatch")
                if any(type(demo_id) is not int or catalogs["train"].get(demo_id) != family
                       for demo_id, family in zip(demo_ids, demo_families)):
                    raise ValueError("demonstration row/source family differs from training catalog")
                demos.update(demo_families)
            rows.extend(batch_rows)
            families.update(view["source_family_ids"])
        if ordered_views != list(by_view):
            raise ValueError("producer view inventory differs from ordered prepared stream")
        role_rows[role], role_families[role], role_demos[role] = rows, families, demos
        specs[role] = spec
    if (not set(role_rows["optimization"]) <= set(ids["train_rows"])
            or role_rows["objective_validation"] != ids["objective_rows"]
            or not set(role_rows["geometry_training"]) <= set(ids["train_rows"])
            or not set(role_rows["geometry_selection"]) <= set(ids["validation_rows"])):
        raise ValueError("actual batch rows violate declared optimization/selection roles")
    if manifest.get("protocol_id") in NATIVE16_PROTOCOLS:
        from .presentation import PresentationSampler
        stream = manifest.get("presentation_stream")
        expected = {"policy": "epoch-permutations-v1", "seed": 20260920,
                    "total_presentations": manifest["q"] * 16, "start_presentation": 0,
                    "pool_identity": canonical_digest(ids["train_rows"])}
        if stream != expected:
            raise ValueError("native16 requires explicit complete-pool fresh presentation stream")
        sampler = PresentationSampler(ids["train_rows"], manifest["q"] * 16, seed=stream["seed"])
        if role_rows["optimization"] != sampler.actual_ids.tolist():
            raise ValueError("native16 ordered rows differ from complete-pool epoch permutations")
    allowed = set(specs["optimization"]["data"]["source_family_ids"])
    if allowed != set(catalogs["train"].values()):
        raise ValueError("optimization producer pool differs from complete training family catalog")
    forbidden = set(catalogs["validation"].values()) | set(task_template["source_family_ids"])
    if allowed & forbidden or role_families["optimization"] & forbidden:
        raise ValueError("optimization pool overlaps selection/development source families")
    if not role_families["geometry_training"] <= allowed:
        raise ValueError("geometry training families must belong to optimization pool")
    if any(demos & forbidden or not demos <= allowed for demos in role_demos.values()):
        raise ValueError("demonstrations must use the optimization pool and exclude heldout families")


# Transferred from encoder_stages_evaluation; serialized meanings remain unchanged.
STAGES_CONDITIONS = ('no_noise', -6, 0, 6, 18)


def open_stages_terminal(training_run, metadata, plan):
    root = Path(training_run).resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    freeze_path = root / 'freeze.json'
    if (campaign['status'] != 'training_frozen' or campaign['completed_updates'] != plan.horizon
            or campaign['prepared_identity'] != metadata['prepared_identity']
            or campaign['freeze_reference'] != {'path': str(freeze_path), 'sha256': file_digest(freeze_path)}):
        raise ValueError('own training trajectory is not completely frozen')
    freeze = json.loads(freeze_path.read_text())
    if (freeze['status'] != 'complete' or freeze['arm'] != plan.arm or freeze['width'] != plan.width
            or freeze['final_panels'] != [[site, condition] for condition in STAGES_CONDITIONS for site in PILOT_SITES]
            or file_digest(root / 'metrics.jsonl') != freeze['metrics']['sha256']):
        raise ValueError('terminal study plan or scalar log differs')
    expected = {k: v for k, v in metadata.items() if k not in {'phase', 'lineage'}}
    if plan.arm != 'A':
        reset = campaign['phase_reset']
        parent_ref = CheckpointRef(**reset['parent'])
        if Path(parent_ref.path).resolve() != root / f'step_{plan.horizon // 2:06d}.pt':
            raise ValueError('phase reset parent path differs from own training run')
        parent = open_state(parent_ref, expected=metadata)
        if parent.payload['metadata']['completed_updates'] != plan.horizon // 2:
            raise ValueError('phase reset parent did not complete stage one')
        expected['parent_identity'] = parent_ref.sha256
    state = open_state(CheckpointRef(**freeze['checkpoint']), expected=expected)
    if (state.payload['metadata']['completed_updates'] != plan.horizon
            or state.payload['metadata']['sharing']['arm'] != plan.arm
            or state.payload['metadata']['sharing']['width'] != plan.width):
        raise ValueError('terminal checkpoint differs from own arm/width')
    return state, campaign['freeze_reference']


INITIAL_CONDITIONS = ('no_noise', -6)


SUPPLEMENTAL_CONDITIONS = (0, 6, 18)


def _reopen_original13(state):
    reopened = open_state(state.reference, expected=state.payload['metadata'])
    if reopened.payload['kind'] != 'original13_v1' or not state_exact(reopened.payload, state.payload):
        raise ValueError('original13 validated checkpoint payload differs')
    return reopened, Original13Plan.from_resolved(reopened.payload['metadata']['original13_plan'])


def _open_bound_reference(reference):
    # Discover identities only from the explicitly selected trusted local bytes.
    # Central open_state still validates the full envelope and those identities.
    data = Path(reference.path).read_bytes()
    if len(data) != reference.size or hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError('checkpoint bytes incomplete or corrupt')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    return open_state(reference, expected=payload['metadata'])


def open_original13_terminal(training_run):
    """Open the sole terminal in this frozen segment; never search best/latest."""
    root = Path(training_run).resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    freeze_path = root / 'freeze.json'
    reference = {'path': str(freeze_path), 'sha256': file_digest(freeze_path)}
    if (campaign['schema'] != 'original13-training-v1' or campaign['status'] != 'training_frozen'
            or campaign['freeze_reference'] != reference):
        raise ValueError('original13 own frozen training run required')
    freeze = json.loads(freeze_path.read_text())
    state = _open_bound_reference(CheckpointRef(**freeze['checkpoint']))
    state, plan = _reopen_original13(state)
    terminal = [c for c in campaign['checkpoints'] if c['role'] == 'terminal']
    if (freeze['schema'] != 'original13-freeze-v1' or freeze['status'] != 'training_frozen'
            or campaign['plan_identity'] != plan.identity or freeze['plan_identity'] != plan.identity
            or campaign['completed_updates'] != plan.horizon
            or state.payload['metadata']['completed_updates'] != plan.horizon
            or state.payload['metadata']['phase_start_step'] != plan.reconstruction_stop
            or len(terminal) != 1
            or {k: terminal[0][k] for k in ('path', 'sha256', 'size')} != asdict(state.reference)
            or Path(state.reference.path).resolve().parent != root
            or Path(freeze['metrics']['path']).resolve() != root / 'metrics.jsonl'
            or file_digest(root / 'metrics.jsonl') != freeze['metrics']['sha256']):
        raise ValueError('original13 terminal, plan, endpoint or scalar bytes differ')
    return state, reference


def _validate_template(template, count):
    required = ('task', 'input_ids', 'source_family_ids', 'settings', 'noise',
                'data_settings', 'scorer', 'precision', 'backend', 'reference_corpus',
                'prompt_policy', 'layout')
    if any(key not in template for key in required):
        raise ValueError('original13 scoring template is incomplete')
    settings = template['settings']
    if (template['task'] != 'hellaswag' or template['expected_items'] != count
            or template['input_ids'] != list(range(count))
            or len(template['source_family_ids']) != count
            or any(not isinstance(f, str) or not f for f in template['source_family_ids'])
            or settings['num_samples'] != count or settings['num_fewshot'] != 5
            or settings['batch_size'] != 16 or settings['scoring_policy'] != 'fp32-v1'
            or settings.get('numerical_policy', 'native') != 'native'
            or settings.get('vanilla') is not False
            or template['scorer'] != 'lm-eval-hellaswag-fp32-v1-acc-and-acc_norm'
            or template['reference_corpus']['split'] != 'validation'
            or type(template['noise']['seed']) is not int or not template['noise']['namespace']):
        raise ValueError('original13 fixed full/five-shot/candidate16/FP32 recipe differs')


def terminal_original13_request(training_run, template, *, scope='initial'):
    """Bind an explicit scope; supporting supplements does not request them."""
    if scope not in ('initial', 'supplemental'):
        raise ValueError('unknown original13 terminal request scope')
    state, freeze = open_original13_terminal(training_run)
    _, plan = _reopen_original13(state)
    _validate_template(template, 10042)
    details = state.payload['metadata']
    return {**copy.deepcopy(template), 'schema': 'original13-terminal-request-v1',
            'scope': scope, 'training_run': str(Path(training_run).resolve()),
            'freeze_reference': freeze, 'checkpoint': asdict(state.reference),
            'checkpoint_sha256': state.reference.sha256, 'kind': 'original13_v1',
            'plan_identity': plan.identity, 'input_identity': plan.input_identity,
            'source': details['source_identity'], 'config': details['config_identity'],
            'data': details['data_identity'], 'protocol': details['protocol_identity'],
            'prepared_identity': details['prepared_identity'], 'step': plan.horizon,
            'width': plan.width, 'ratio': plan.ratio, 'sites': list(ORIGINAL13_SITES),
            'conditions': list(INITIAL_CONDITIONS if scope == 'initial' else SUPPLEMENTAL_CONDITIONS)}


def _validate_request(state, request, *, observation):
    state, plan = _reopen_original13(state)
    details = state.payload['metadata']
    if (request['plan_identity'] != plan.identity
            or request['checkpoint_sha256'] != state.reference.sha256
            or request['step'] != details['completed_updates']):
        raise ValueError('original13 request checkpoint/plan/step differs')
    if observation:
        _validate_template(request, 256)
        panel = request['panel']
        if (request['schema'] != 'original13-observation-request-v1'
                or canonical_digest(panel) != plan.observation_identity
                or (request['step'], request['state_role']) not in plan.observations
                or request['sites'] != list(OBSERVATION_SITES)
                or request['conditions'] != list(INITIAL_CONDITIONS)
                or request['input_ids'] != panel['input_ids']
                or request['source_family_ids'] != panel['source_family_ids']
                or request['noise'] != panel['noise']
                or (request['state_role'] == 'pre_reset' and details['phase_start_step'] != 0)
                or (request['state_role'] == 'functional_observation'
                    and details['phase_start_step'] != plan.reconstruction_stop)):
            raise ValueError('original13 fixed observation panel/endpoint differs')
    else:
        rebound = terminal_original13_request(request['training_run'], request, scope=request['scope'])
        if request != rebound or request['kind'] != 'original13_v1':
            raise ValueError('original13 terminal request differs from own frozen run')
    return state, plan


def original13_training_rows(training_run):
    """Preserve raw/weighted stream measures and denominators; never smooth/select.

    A continued segment exposes only its own rows and the prior-segment pointers.
    The verified terminal exposure ledger remains the cumulative authority.
    """
    from .experiment_records import read_events
    root = Path(training_run).resolve()
    state, _ = open_original13_terminal(root)
    campaign = json.loads((root / 'campaign.json').read_text())
    rows = []
    for event in read_events(root / 'metrics.jsonl'):
        if event['event_type'] != 'update':
            continue
        payload = event['payload']
        if payload['nonfinite'] or payload['skipped']:
            raise ValueError('original13 report contains unsuccessful scalar update')
        rows.append({'global_update': payload['completed_step'], 'phase': event['phase_id'],
                     'site': payload['site_id'], 'objective': copy.deepcopy(payload['objective']),
                     'exposure': copy.deepcopy(payload['exposure']), 'lr_used': payload['lr_used'],
                     'cost': {'timing': copy.deepcopy(payload.get('timing', {})),
                              'memory': copy.deepcopy(payload.get('memory', {}))}})
    count = campaign['segment_updates']
    if [r['global_update'] for r in rows] != list(range(campaign['completed_updates']-count+1,
                                                       campaign['completed_updates']+1)):
        raise ValueError('original13 segment scalar cadence differs')
    return {'rows': rows, 'segment_updates': count,
            'segment_ancestry': state.payload['metadata']['segment_ancestry'],
            'cumulative_exposure': state.payload['metadata']['exposure'],
            'metrics_sha256': file_digest(root / 'metrics.jsonl')}


def original13_report(training_run, *, initial=None, supplemental=None, observations=(),
                      initial_request=None, supplemental_request=None):
    """Offline descriptive accounting from requests and durable receipts only."""
    state, _ = open_original13_terminal(training_run)
    _, plan = _reopen_original13(state)
    campaign = json.loads((Path(training_run) / 'campaign.json').read_text())
    report = {'schema': 'original13-report-v1', 'width': plan.width, 'ratio': plan.ratio,
              'checkpoint': asdict(state.reference), 'training': campaign['status'],
              'observations': copy.deepcopy(campaign['observations']), 'terminal': {},
              'collection': 'complete', 'results': [],
              'training_scalars': original13_training_rows(training_run),
              'limits': 'Descriptive single-seed development evidence; common seeds do not prove equal realized AWGN. '
                        'Old six-encoder runs are contextual. No superiority/convergence claim.'}
    for scope, root, requested in (('initial', initial, initial_request),
                                   ('supplemental', supplemental, supplemental_request)):
        if requested is not None:
            _validate_request(state, requested, observation=False)
            if requested['scope'] != scope:
                raise ValueError('original13 report explicit request scope differs')
        if root is None and requested is not None:
            count = len(requested['sites']) * len(requested['conditions'])
            report['terminal'][scope] = {'status': 'pending', 'requested': count, 'completed': 0,
                                        'failed': 0, 'pending': count}
            continue
        if root is None:
            report['terminal'][scope] = {'status': 'not_requested', 'requested': 0, 'completed': 0,
                                        'failed': 0, 'pending': 0}
            continue
        receipt = json.loads((Path(root) / 'result.json').read_text())
        _verify_original13_result(receipt, state, observation=False)
        if (receipt['request']['scope'] != scope
                or (requested is not None and requested != receipt['request'])):
            raise ValueError('original13 report request scope differs')
        report['terminal'][scope] = _panel_counts(receipt)
        report['results'].extend(copy.deepcopy(receipt['panels']))
    endpoints = set()
    for root in observations:
        receipt = json.loads((Path(root) / 'result.json').read_text())
        observed_state = _open_bound_reference(CheckpointRef(**receipt['checkpoint']))
        _verify_original13_result(receipt, observed_state, observation=True)
        endpoint = (receipt['request']['step'], receipt['request']['state_role'])
        if endpoint in endpoints:
            raise ValueError('original13 report duplicate observation endpoint')
        endpoints.add(endpoint)
        if (receipt['plan_identity'] != plan.identity
                or not _own_observation_checkpoint(state, campaign, receipt['checkpoint'], receipt['request'])):
            raise ValueError('original13 report observation trajectory differs')
        report.setdefault('observation_receipts', []).append(_panel_counts(receipt))
    report['observations_status'] = campaign['observations_status']
    return report


def _own_observation_checkpoint(terminal, campaign, reference, request):
    """Endpoint must be saved by this segment or an explicitly carried ancestor.

    Continuation keeps an earlier failed/pending observation pending; a supplied
    receipt is descriptive evidence, not an automatic training-status rewrite.
    """
    details = terminal.payload['metadata']
    campaigns = [(campaign, details['completed_updates'])]
    for parent_reference in details['segment_ancestry']:
        parent = _open_bound_reference(CheckpointRef(**parent_reference))
        fields = ('source_identity', 'config_identity', 'prepared_identity', 'plan_identity',
                  'input_identity', 'initialization_identity', 'stream_identity')
        if any(parent.payload['metadata'][key] != details[key] for key in fields):
            raise ValueError('original13 report ancestor identity differs')
        parent_campaign = json.loads((Path(parent_reference['path']).parent / 'campaign.json').read_text())
        if not any({k: saved[k] for k in ('path', 'sha256', 'size')} == parent_reference
                   for saved in parent_campaign['checkpoints']):
            raise ValueError('original13 ancestor campaign does not own carried checkpoint')
        campaigns.append((parent_campaign, parent.payload['metadata']['completed_updates']))
    for owner, inherited_through in campaigns:
        if request['step'] > inherited_through:
            continue
        matches = [saved for saved in owner['checkpoints'] if saved['role'] == request['state_role']
                   and {k: saved[k] for k in ('path', 'sha256', 'size')} == reference]
        observation = next((row for row in owner['observations']
                            if (row['step'], row['state_role']) == (request['step'], request['state_role'])), None)
        if matches and observation and observation.get('request_identity') == canonical_digest(request):
            return True
    return False


def _panel_counts(receipt):
    return {'status': receipt['status'], 'requested': len(receipt['panels']),
            'completed': sum(p['status'] == 'complete' for p in receipt['panels']),
            'failed': sum(p['status'] == 'failed' for p in receipt['panels']),
            'pending': sum(p['status'] == 'pending' for p in receipt['panels']),
            'requested_items': sum(p['requested_items'] for p in receipt['panels']),
            'observed_items': sum(p['completed_items'] for p in receipt['panels']),
            'elapsed_seconds': receipt['elapsed_seconds']}


def _verify_original13_result(receipt, state, *, observation):
    _validate_request(state, receipt['request'], observation=observation)
    if receipt['schema'] != 'original13-evaluation-receipt-v1':
        raise ValueError('original13 result schema differs')
    _verify_panel_result(receipt, state)


def _verify_panel_result(receipt, state):
    if (receipt['request_identity'] != canonical_digest(receipt['request'])
            or receipt['checkpoint'] != asdict(state.reference)
            or receipt['plan_identity'] != state.payload['metadata']['plan_identity']):
        raise ValueError('original13 result identity differs')
    if receipt['status'] not in ('complete', 'failed', 'incomplete'):
        raise ValueError('unsupported evaluation receipt status')
    expected = [(site, condition) for condition in receipt['request']['conditions']
                for site in receipt['request']['sites']]
    if [(p['site'], p['condition']) for p in receipt['panels']] != expected:
        raise ValueError('original13 result panels missing/duplicated/reordered')
    for panel in receipt['panels']:
        if panel['status'] not in ('complete', 'pending', 'failed'):
            raise ValueError('original13 unsupported panel status')
        if (type(panel['requested_items']) is not int
                or panel['requested_items'] != receipt['request']['expected_items']
                or type(panel['completed_items']) is not int
                or not 0 <= panel['completed_items'] <= panel['requested_items']
                or (panel['status'] != 'complete' and panel['completed_items'] != 0)):
            raise ValueError('evaluation panel requested/completed denominator differs')
        if panel['status'] == 'complete':
            panel_request = {**receipt['request'], 'schema': ('recipe-panel-request-v1'
                if receipt['schema'] == 'recipe-evaluation-receipt-v1' else 'original13-panel-request-v1'),
                             'target_site': panel['site'], 'conditions': [panel['condition']]}
            if (panel['panel_identity'] != canonical_digest(panel_request)
                    or panel['completed_items'] != panel['requested_items']
                    or panel['denominator'] != panel['requested_items']):
                raise ValueError('original13 panel identity/completed denominator differs')
            path = Path(panel['receipt']['path'])
            if file_digest(path) != panel['receipt']['sha256']:
                raise ValueError('original13 panel receipt bytes differ')
            if json.loads((path.parent / 'request.json').read_text()) != panel_request:
                raise ValueError('original13 panel request differs')
            saved = json.loads(path.read_text())
            if saved != {k: v for k, v in panel.items() if k != 'receipt'}:
                raise ValueError('original13 panel summary differs from receipt')
            for name, checksum in saved['output_hashes'].items():
                if Path(name).name != name or file_digest(path.parent / name) != checksum:
                    raise ValueError('original13 panel artifact bytes differ')
    if receipt['status'] == 'failed' and not any(p['status'] == 'failed' for p in receipt['panels']):
        raise ValueError('failed evaluation receipt requires a failed panel')
    if (receipt['status'] == 'complete') != all(p['status'] == 'complete' for p in receipt['panels']):
        raise ValueError('original13 completion differs from actual panel receipts')


# Transferred from encoder_pilot_plan; serialized meanings remain unchanged.
PILOT_SITES = ('enc_emb', 'enc_l4', 'enc_l9', 'enc_l14', 'enc_l19', 'enc_fn')


PILOT_SITE_SPECS = {name: ({'stack': 'enc', 'where': 'after_embed'} if name == 'enc_emb' else
                    {'stack': 'enc', 'where': 'after_final_norm'} if name == 'enc_fn' else
                    {'stack': 'enc', 'where': 'after_layer', 'index': int(name[5:])})
              for name in PILOT_SITES}


@dataclass(frozen=True)
class EncoderPilotPlan:
    views: tuple[BatchView, ...]
    study_pairing_id: str
    synthetic: bool = False
    protocol_id: str = ENCODER_PILOT_PROTOCOL
    trained_sites: tuple[str, ...] = PILOT_SITES

    def __post_init__(self):
        if (self.protocol_id != ENCODER_PILOT_PROTOCOL or self.trained_sites != PILOT_SITES
                or type(self.synthetic) is not bool or not self.study_pairing_id
                or not isinstance(self.views, tuple) or self.q not in ((4,) if self.synthetic else (400, 6400))
                or any(v.sequences != 16 for v in self.views)
                or len({v.view_sha256 for v in self.views}) != self.q):
            raise ValueError('pilot requires original six sites and distinct native16 q400/q6400 views (CPU q4)')

    @property
    def q(self):
        return len(self.views)

    @property
    def batch_size(self):
        return 16

    @property
    def name(self):
        return 'shared'

    @property
    def plan(self):
        return self

    @property
    def horizon(self):
        return 6 * self.q

    @property
    def checkpoint_steps(self):
        if self.q == 6400:
            return (0, 9600, 19200, 28800, 38400)
        return self.objective_steps

    @property
    def objective_steps(self):
        return (0, self.horizon // 2, self.horizon)

    def update_at(self, index):
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError('update outside fixed six-site pilot horizon')
        local, position = divmod(index, 6)
        site, view = PILOT_SITES[position], self.views[local]
        key = NoiseKey(self.study_pairing_id, 'training', site, local, view.view_sha256,
                       view.layout_sha256, condition='uniform', draw_schema='runtime-awgn-replay-v2')
        return ScheduledUpdate('shared', index, site, local, view, key,
                               sharing_lr_factor(index, self.horizon))


def build_pilot_state(plan, completed):
    if type(completed) is not int or not 0 <= completed <= plan.horizon:
        raise ValueError('pilot completed count outside horizon')
    counts = {site: completed // 6 + (position < completed % 6) for position, site in enumerate(PILOT_SITES)}
    vectors = {field: [getattr(view, attr) for view in plan.views] for field, attr in
               [('source_valid_per_view', 'source_tokens'), ('target_valid_per_view', 'target_tokens'),
                ('padded_per_view', 'padded_tokens')]}
    next_site = PILOT_SITES[completed % 6] if completed < plan.horizon else None
    sharing = {'schema': 'encoder-original6-pilot-state-v1', 'synthetic': plan.synthetic,
               'learner': 'shared', 'q': plan.q, 'batch_size': 16, 'trained_sites': list(PILOT_SITES),
               'horizon': plan.horizon, 'protocol_id': ENCODER_PILOT_PROTOCOL,
               'study_pairing_id': plan.study_pairing_id,
               'ordered_views': [view.__dict__ for view in plan.views], **vectors,
               'completed_per_site': counts, 'next_site': next_site}
    for label, field in [('source_valid_per_site', 'source_valid_per_view'),
                         ('target_valid_per_site', 'target_valid_per_view'), ('padded_per_site', 'padded_per_view')]:
        sharing[label] = {site: sum(vectors[field][:count]) for site, count in counts.items()}
    return sharing, {'completed_updates': completed, 'offset': completed * 16,
                     'view_offsets': counts, 'next_site': next_site}


def validate_pilot_state(payload):
    metadata = payload['metadata']
    sharing = metadata['sharing']
    plan = EncoderPilotPlan(tuple(BatchView(**v) for v in sharing['ordered_views']),
                            sharing['study_pairing_id'], synthetic=sharing['synthetic'])
    expected, stream = build_pilot_state(plan, metadata['completed_updates'])
    if (payload['schema'] != 'experiment-state-v2' or payload['kind'] != 'shared_encoder_v2'
            or metadata['protocol_identity'] != ENCODER_PILOT_PROTOCOL or metadata['phase'] != 'both'
            or canonical_digest(sharing) != canonical_digest(expected) or payload['stream'] != stream
            or type(payload['scheduler']['last_epoch']) is not int
            or payload['scheduler']['last_epoch'] != metadata['completed_updates']):
        raise ValueError('pilot state counters, protocol or horizon mismatch')
    declarations = metadata['evaluation_sites']
    if set(declarations) != set(PILOT_SITES):
        raise ValueError('pilot requires exactly six declared trained sites')
    topologies = set()
    for name, declaration in declarations.items():
        site = ResolvedSite(**declaration['site'])
        path = ('encoder.layers.0.input' if name == 'enc_emb' else 'encoder.norm' if name == 'enc_fn'
                else f'encoder.layers.{site.index}')
        if (declaration['role'] != 'trained' or site.site_id != name or site.split != PILOT_SITE_SPECS[name]
                or site.module_path != path or site.layer_count < 20 or site.hidden_dim <= 0
                or site.model_revision != metadata['model_revision']):
            raise ValueError('pilot resolved site declaration mismatch')
        topologies.add((site.model_revision, site.layer_count, site.hidden_dim))
    if len(topologies) != 1:
        raise ValueError('pilot sites have inconsistent backbone topology')


def final_panels(q=400):
    """Long study: six no-noise only; short pilot retains its original panels."""
    return tuple((site, condition) for condition in (('no_noise',) if q == 6400 else CONDITIONS) for site in PILOT_SITES)


# Transferred from encoder_stages_plan; serialized meanings remain unchanged.
ENCODER_STAGES_PROTOCOL = 'encoder-original6-native16-stages-v1'


WIDTHS = (512, 1152, 2304)


ARMS = ('A', 'E', 'R')


PEAK_LEARNING_RATE = 2e-4


TOTAL_SCIENTIFIC_UPDATES = len(WIDTHS) * len(ARMS) * 60000


@dataclass(frozen=True)
class EncoderStage:
    phase_id: str
    objective_kind: str
    start: int
    stop: int
    reset_optimizer: bool = False

    @property
    def horizon(self):
        return self.stop - self.start

    @property
    def objective_weights(self):
        return {'kl': 0.0 if self.objective_kind == 'reconstruction' else 1.0,
                'hidden': 0.1, 'memory': 0.0}

    @property
    def warmup_updates(self):
        return max(1, self.horizon // 20)


@dataclass(frozen=True)
class EncoderStagesPlan:
    views: tuple[BatchView, ...]
    study_pairing_id: str
    arm: str
    synthetic: bool = False
    width: int = 512
    protocol_id: str = ENCODER_STAGES_PROTOCOL
    trained_sites: tuple[str, ...] = PILOT_SITES

    def __post_init__(self):
        if (self.protocol_id != ENCODER_STAGES_PROTOCOL or self.trained_sites != PILOT_SITES
                or self.arm not in ARMS or type(self.synthetic) is not bool
                or not isinstance(self.study_pairing_id, str) or not self.study_pairing_id.strip()
                or not isinstance(self.views, tuple) or self.q < 2 or self.q % 2
                or (not self.synthetic and self.q != 10000)
                or any(not isinstance(v, BatchView) or v.sequences != 16 for v in self.views)
                or len({v.view_sha256 for v in self.views}) != self.q
                or type(self.width) is not int or self.width <= 0
                or (not self.synthetic and self.width not in WIDTHS)):
            raise ValueError('stages require six sites, A/E/R, distinct native16 q10000 views and approved width (CPU even q)')

    @property
    def q(self):
        return len(self.views)

    @property
    def batch_size(self):
        return 16

    @property
    def accumulation(self):
        return 1

    @property
    def name(self):
        return 'shared'

    @property
    def plan(self):
        return self

    @property
    def horizon(self):
        return 6 * self.q

    @property
    def boundary_steps(self):
        return (0, self.horizon // 2, self.horizon)

    @property
    def reset_steps(self):
        return () if self.arm == 'A' else (self.horizon // 2,)

    @property
    def stages(self) -> tuple[EncoderStage, ...]:
        if self.arm == 'A':
            return (EncoderStage('phase1', 'combined', 0, self.horizon),)
        midpoint = self.horizon // 2
        return (EncoderStage('phase1', 'reconstruction' if self.arm == 'R' else 'combined', 0, midpoint),
                EncoderStage('phase2', 'combined', midpoint, self.horizon, True))

    @property
    def checkpoint_steps(self):
        interval = 6 * max(1, self.q // 10)
        return tuple(sorted(set(range(0, self.horizon + 1, interval)) | set(self.boundary_steps)))

    @property
    def objective_steps(self):
        return ()

    @property
    def observations(self):
        return ()

    def stage_at(self, index):
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError('update outside fixed six-site stages horizon')
        return self.stages[0] if self.arm == 'A' or index < self.horizon // 2 else self.stages[1]

    def objective_kind_at(self, index):
        return self.stage_at(index).objective_kind

    def scheduler_index_at(self, index):
        return index - self.stage_at(index).start

    def scheduler_horizon_at(self, index):
        return self.stage_at(index).horizon

    def update_at(self, index):
        stage = self.stage_at(index)
        local, position = divmod(index, 6)
        site, view = PILOT_SITES[position], self.views[local]
        key = NoiseKey(self.study_pairing_id, 'training', site, local, view.view_sha256,
                       view.layout_sha256, condition='uniform', draw_schema='runtime-awgn-replay-v2')
        return ScheduledUpdate('shared', index, site, local, view, key,
                               sharing_lr_factor(index - stage.start, stage.horizon))


def build_stages_state(plan, completed, *, phase_start_step=None):
    """Retain global stream/exposure while identifying either side of a reset.

    At the midpoint, start zero denotes the saved phase-one terminal checkpoint;
    start midpoint denotes the distinct fresh-optimizer phase-two checkpoint.
    """
    if type(completed) is not int or not 0 <= completed <= plan.horizon:
        raise ValueError('stages completed count outside horizon')
    if phase_start_step is None:
        phase_start_step = plan.stages[-1].start if completed >= plan.horizon // 2 else 0
    matches = [stage for stage in plan.stages if stage.start == phase_start_step
               and stage.start <= completed <= stage.stop]
    if type(phase_start_step) is not int or len(matches) != 1:
        raise ValueError('invalid phase start or completed count across reset boundary')
    stage = matches[0]
    counts = {site: completed // 6 + (position < completed % 6) for position, site in enumerate(PILOT_SITES)}
    vectors = {field: [getattr(view, attr) for view in plan.views] for field, attr in
               [('source_valid_per_view', 'source_tokens'), ('target_valid_per_view', 'target_tokens'),
                ('padded_per_view', 'padded_tokens')]}
    next_site = PILOT_SITES[completed % 6] if completed < plan.horizon else None
    sharing = {'schema': 'encoder-original6-stages-state-v1', 'synthetic': plan.synthetic,
               'learner': 'shared', 'arm': plan.arm, 'width': plan.width,
               'q': plan.q, 'batch_size': 16, 'accumulation': 1, 'trained_sites': list(PILOT_SITES),
               'horizon': plan.horizon, 'protocol_id': ENCODER_STAGES_PROTOCOL,
               'study_pairing_id': plan.study_pairing_id, 'completed_updates': completed,
               'phase_id': stage.phase_id, 'phase_start_step': stage.start, 'phase_stop_step': stage.stop,
               'phase_local_completed': completed - stage.start,
               'objective_kind': stage.objective_kind, 'optimizer_reset_step': stage.start if stage.reset_optimizer else None,
               'ordered_views': [view.__dict__ for view in plan.views], **vectors,
               'completed_per_site': counts, 'next_site': next_site}
    for label, field in [('source_valid_per_site', 'source_valid_per_view'),
                         ('target_valid_per_site', 'target_valid_per_view'), ('padded_per_site', 'padded_per_view')]:
        sharing[label] = {site: sum(vectors[field][:count]) for site, count in counts.items()}
    return sharing, {'completed_updates': completed, 'offset': completed * 16,
                     'view_offsets': counts, 'next_site': next_site}


def validate_stages_state(payload):
    metadata = payload['metadata']
    sharing = metadata['sharing']
    plan = EncoderStagesPlan(tuple(BatchView(**v) for v in sharing['ordered_views']),
                             sharing['study_pairing_id'], sharing['arm'],
                             synthetic=sharing['synthetic'], width=sharing['width'])
    expected, stream = build_stages_state(plan, metadata['completed_updates'],
                                          phase_start_step=sharing['phase_start_step'])
    local = expected['phase_local_completed']
    if (payload['schema'] != 'experiment-state-v2' or payload['kind'] != 'shared_encoder_v2'
            or metadata['protocol_identity'] != ENCODER_STAGES_PROTOCOL
            or metadata['phase'] != expected['phase_id']
            or canonical_digest(sharing) != canonical_digest(expected) or payload['stream'] != stream
            or type(payload['scheduler']['last_epoch']) is not int or payload['scheduler']['last_epoch'] != local):
        raise ValueError('stages state counters, phase, protocol or horizon mismatch')
    moments = payload['optimizer'].get('state')
    if not isinstance(moments, dict) or (not moments) != (local == 0):
        raise ValueError('stages optimizer moments mismatch phase-local updates')
    for state in moments.values():
        try:
            step = float(state['step'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('invalid stages optimizer step') from error
        if not math.isfinite(step) or step != local:
            raise ValueError('stages optimizer step mismatch phase-local updates')
    declarations = metadata['evaluation_sites']
    if set(declarations) != set(PILOT_SITES):
        raise ValueError('stages requires exactly six declared trained sites')
    topologies = set()
    for name, declaration in declarations.items():
        site = ResolvedSite(**declaration['site'])
        path = ('encoder.layers.0.input' if name == 'enc_emb' else 'encoder.norm' if name == 'enc_fn'
                else f'encoder.layers.{site.index}')
        if (declaration['role'] != 'trained' or site.site_id != name or site.split != PILOT_SITE_SPECS[name]
                or site.module_path != path or type(site.layer_count) is not int or site.layer_count < 20
                or type(site.hidden_dim) is not int or site.hidden_dim <= 0
                or not site.model_revision or site.model_revision != metadata['model_revision']):
            raise ValueError('stages resolved site declaration mismatch')
        topologies.add((site.model_revision, site.layer_count, site.hidden_dim))
    if len(topologies) != 1:
        raise ValueError('stages sites have inconsistent backbone topology')


ORIGINAL13_PROTOCOL = 'original13-native16-ratios-v1'


ORIGINAL13_SITES = ('enc_emb', 'enc_l4', 'enc_l9', 'enc_l14', 'enc_l19', 'enc_fn',
                    'dec_l0', 'dec_l4', 'dec_l8', 'dec_l12', 'dec_l16', 'dec_l20', 'dec_l24')


ORIGINAL13_RATIOS = (50, 75, 90)


OBSERVATION_SITES = ('enc_l9', 'enc_fn', 'dec_l0', 'dec_l24')


@dataclass(frozen=True)
class Original13Plan:
    views: tuple[BatchView, ...]
    study_pairing_id: str
    ratio: int
    width: int
    input_identity: str
    observation_identity: str
    synthetic: bool = False

    def __post_init__(self):
        if (type(self.ratio) is not int or self.ratio not in ORIGINAL13_RATIOS
                or type(self.width) is not int or self.width <= 0
                or type(self.synthetic) is not bool or not isinstance(self.views, tuple)
                or not self.views or self.q % 400
                or (not self.synthetic and (self.q != 10000 or self.width not in WIDTHS))
                or any(not isinstance(v, BatchView) or v.sequences != 16 for v in self.views)
                or len({v.view_sha256 for v in self.views}) != self.q
                or not isinstance(self.study_pairing_id, str) or not self.study_pairing_id.strip()):
            raise ValueError('original13 requires fixed ratios, width and distinct native16 complete-quarter views')
        for identity in (self.input_identity, self.observation_identity):
            if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
                raise ValueError('original13 input and observation require SHA256 identities')

    @property
    def q(self):
        return len(self.views)

    @property
    def plan(self):
        return self

    @property
    def name(self):
        return 'shared'

    @property
    def protocol_id(self):
        return ORIGINAL13_PROTOCOL

    @property
    def trained_sites(self):
        return ORIGINAL13_SITES

    @property
    def batch_size(self):
        return 16

    @property
    def accumulation(self):
        return 1

    @property
    def horizon(self):
        return 13 * self.q

    @property
    def reconstruction_stop(self):
        return self.horizon * self.ratio // 100

    @property
    def stages(self):
        stop = self.reconstruction_stop
        return (EncoderStage('reconstruction', 'reconstruction', 0, stop),
                EncoderStage('functional', 'combined', stop, self.horizon, True))

    @property
    def checkpoint_states(self):
        stop = self.reconstruction_stop
        functional = self.horizon - stop
        states = [(0, 'initialization')]
        states += [(stop * n // 4, 'pre_reset' if n == 4 else 'reconstruction') for n in range(1, 5)]
        states += [(stop, 'post_reset'), (stop + functional // 10, 'functional_observation')]
        states += [(stop + functional * n // 4, 'terminal' if n == 4 else 'functional') for n in range(1, 5)]
        return tuple(states)

    @property
    def observations(self):
        return ((self.reconstruction_stop, 'pre_reset'),
                (self.reconstruction_stop + (self.horizon - self.reconstruction_stop) // 10,
                 'functional_observation'))

    def stage_at(self, index):
        if type(index) is not int or not 0 <= index < self.horizon:
            raise ValueError('update outside original13 horizon')
        return self.stages[int(index >= self.reconstruction_stop)]

    def objective_at(self, index):
        stage = self.stage_at(index)
        decoder = ORIGINAL13_SITES[index % 13].startswith('dec_')
        return {'kl': float(stage.objective_kind == 'combined'),
                'hidden': .05 if decoder else .1, 'memory': .05 if decoder else 0.0}

    def update_at(self, index):
        stage = self.stage_at(index)
        local, position = divmod(index, 13)
        view, site = self.views[local], ORIGINAL13_SITES[position]
        key = NoiseKey(self.study_pairing_id, stage.objective_kind, site, local,
                       view.view_sha256, view.layout_sha256, condition='uniform',
                       draw_schema='original13-role-awgn-v1')
        return ScheduledUpdate('shared', index, site, local, view, key,
                               sharing_lr_factor(index - stage.start, stage.horizon))

    def resolved(self):
        """Versioned immutable plan; ratios never alter initializer/input identity."""
        return {'schema': ORIGINAL13_PROTOCOL, 'sites': list(ORIGINAL13_SITES),
                'roles': ['encoder_hidden', 'decoder_hidden', 'receiver_memory'],
                'ratio': self.ratio, 'width': self.width, 'synthetic': self.synthetic,
                'views': [v.__dict__ for v in self.views], 'study_pairing_id': self.study_pairing_id,
                'input_identity': self.input_identity, 'observation_identity': self.observation_identity,
                'batch_size': 16, 'accumulation': 1, 'quota_per_site': self.q, 'horizon': self.horizon,
                'stages': [{**s.__dict__, 'warmup_updates': s.warmup_updates,
                            'peak_lr': 2e-4, 'min_lr_ratio': 0} for s in self.stages],
                'objectives': {'encoder': {'hidden': .1, 'memory': 0},
                               'decoder': {'hidden': .05, 'memory': .05}, 'functional_kl': 1.0,
                               'denominators': {'kl': 'valid_target_tokens', 'hidden': 'valid_hidden_samples',
                                                'memory': 'valid_memory_samples'}},
                'checkpoint_states': [list(s) for s in self.checkpoint_states],
                'observations': [{'step': step, 'state': state, 'panel': self.observation_identity,
                                  'sites': list(OBSERVATION_SITES), 'conditions': ['no_noise', -6]}
                                 for step, state in self.observations],
                'terminal_requests': {'initial': ['no_noise', -6], 'supplemental_supported': [0, 6, 18],
                                      'supplemental_requested': False, 'items': 10042,
                                      'fewshot': 5, 'candidate_batch': 16, 'scoring_policy': 'fp32-v1'}}

    @property
    def identity(self):
        return canonical_digest(self.resolved())

    @classmethod
    def from_resolved(cls, value):
        plan = cls(tuple(BatchView(**v) for v in value['views']), value['study_pairing_id'],
                   value['ratio'], value['width'], value['input_identity'], value['observation_identity'],
                   value['synthetic'])
        if canonical_digest(value) != plan.identity:
            raise ValueError('unsupported or changed original13 plan')
        return plan


def original13_progress(plan, completed, phase_start):
    if type(completed) is not int or type(phase_start) is not int:
        raise ValueError('original13 progress requires integer counts')
    matches = [s for s in plan.stages if s.start == phase_start and s.start <= completed <= s.stop]
    if len(matches) != 1:
        raise ValueError('original13 progress crosses an unrecorded reset')
    phase = matches[0]
    counts = {site: completed // 13 + int(i < completed % 13) for i, site in enumerate(ORIGINAL13_SITES)}
    return {'completed_updates': completed, 'next_site': ORIGINAL13_SITES[completed % 13]
            if completed < plan.horizon else None, 'next_view': completed // 13,
            'offset': completed * 16, 'completed_per_site': counts,
            'phase_id': phase.phase_id, 'phase_start_step': phase.start,
            'phase_local_completed': completed - phase.start, 'reset_count': int(phase.reset_optimizer)}


# Transferred from original13_state; serialized meanings remain unchanged.
def validate_original13_state(payload):


    metadata = payload['metadata']
    plan = Original13Plan.from_resolved(metadata['original13_plan'])
    progress = original13_progress(plan, metadata['completed_updates'], metadata['phase_start_step'])
    if (metadata['protocol_identity'] != ORIGINAL13_PROTOCOL or metadata['plan_identity'] != plan.identity
            or metadata['input_identity'] != plan.input_identity
            or metadata['observation_identity'] != plan.observation_identity
            or metadata['phase'] != progress['phase_id'] or payload['stream'] != progress):
        raise ValueError('original13 state plan, input, phase or stream differs')
    if bool(metadata['parent_identity']) != bool(progress['reset_count']):
        raise ValueError('original13 reset parent missing or unexpected')
    local = progress['phase_local_completed']
    if bool(payload['optimizer'].get('state')) != bool(local):
        raise ValueError('original13 optimizer state differs from phase progress')
    factor = sharing_lr_factor(local, next(s.horizon for s in plan.stages if s.start == metadata['phase_start_step']))
    if any(group['lr'] != 2e-4 * factor for group in payload['optimizer']['param_groups']):
        raise ValueError('original13 LR differs from phase-local schedule')
    groups = payload['optimizer']['param_groups']
    scheduler = payload['scheduler']
    if (scheduler.get('base_lrs') != [2e-4] * len(groups)
            or scheduler.get('_last_lr') != [2e-4 * factor] * len(groups)
            or scheduler.get('_step_count') != local + 1
            or any(g.get('initial_lr') != 2e-4 or g.get('weight_decay') != 0.002500001398541448
                   or g.get('betas') != (.9, .999) or g.get('eps') != 1e-8 or g.get('amsgrad')
                   or g.get('maximize') or g.get('differentiable') for g in groups)
            or not isinstance(metadata['route_state'], dict)):
        raise ValueError('original13 scheduler, Adam recipe or route state differs')
    counts = progress['completed_per_site']
    expected_exposure = {site: {'updates': count, 'sequences': 16 * count,
        **{field: sum(getattr(v, field) for v in plan.views[:count])
           for field in ('source_tokens', 'target_tokens', 'padded_tokens')}} for site, count in counts.items()}
    receipts = metadata['exposure_receipts']
    if (metadata['exposure'] != expected_exposure or metadata['attempted'] != metadata['completed_updates']
            or metadata['valid_tokens'] != sum(v['source_tokens'] for v in expected_exposure.values())
            or not isinstance(receipts, list) or len(receipts) != metadata['completed_updates']
            or any(not isinstance(r, str) or not r for r in receipts) or len(set(receipts)) != len(receipts)
            or not isinstance(metadata['segment_ancestry'], list)
            or metadata['model_state_contract'] != 'original13-canonical-codec-v1'):
        raise ValueError('original13 complete exposure, ancestry or body contract differs')
    if progress['reset_count']:
        entry = metadata['lineage'][-1]
        if (len(metadata['lineage']) != 1 or set(entry) != {
                'parent_sha256', 'parent_checkpoint', 'completed_updates', 'transfer'}
                or entry['completed_updates'] != plan.reconstruction_stop
                or entry['transfer'] != 'weights_only_reset_adam_lr'
                or entry['parent_checkpoint']['sha256'] != metadata['parent_identity']):
            raise ValueError('original13 reset lineage differs')
        _validate_original13_reference(entry['parent_checkpoint'])
    elif metadata['lineage'] != []:
        raise ValueError('original13 reconstruction state cannot have reset lineage')
    for reference in metadata['segment_ancestry']:
        _validate_original13_reference(reference)
    if len({r['sha256'] for r in metadata['segment_ancestry']}) != len(metadata['segment_ancestry']):
        raise ValueError('original13 segment ancestry repeats a checkpoint')


def _validate_original13_reference(reference):
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'size'}
            or not isinstance(reference['path'], str) or not reference['path'].strip()
            or not isinstance(reference['sha256'], str) or len(reference['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in reference['sha256'])
            or type(reference['size']) is not int or reference['size'] <= 0):
        raise ValueError('original13 ancestry requires complete checkpoint references')


# Transferred from sharing_state; serialized meanings remain unchanged.
SHARING_SITES = ("enc_l9", "enc_l19", "enc_fn")


LEARNERS = ("shared", *(f"specialist_{site}" for site in SHARING_SITES))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, label):
    _require(type(value) is int and value >= 0, f"invalid {label}")


def _updates(learner, q, protocol_id=None):
    if learner == "shared":
        if protocol_id == NATIVE16_LONG_PROTOCOL:
            return [SHARING_SITES[index % 3] for index in range(3 * q)]
        return [SHARING_SITES[(sweep + position) % 3] for sweep in range(q) for position in range(3)]
    _require(learner in LEARNERS, "unknown sharing learner")
    return [learner.removeprefix("specialist_")] * q


def build_sharing_state(*, synthetic, learner, ordered_view_identities,
                        completed_updates, source_valid_per_view,
                        target_valid_per_view, padded_per_view, batch_size, protocol_id=None):
    """Build deterministic stream counters from complete effective-batch counts.

The per-view vectors contain one aggregate token count per effective batch,
not individual examples. Their full ordered identities are frozen at step zero.
"""
    _require(type(synthetic) is bool, "synthetic must be explicit boolean")
    for value in (ordered_view_identities, source_valid_per_view, target_valid_per_view, padded_per_view):
        _require(isinstance(value, (list, tuple)), "sharing views/counts must be ordered sequences")
    identities = list(ordered_view_identities)
    native16 = protocol_id in NATIVE16_PROTOCOLS
    q = len(identities) if synthetic or native16 else (400 if protocol_id == DEPTH_PROTOCOL else 200)
    if native16:
        validate_native16_horizon(q)
        _require(synthetic or protocol_id != NATIVE16_LONG_PROTOCOL or q == 6400,
                 "formal native16 long state requires q6400")
        _require(batch_size == 16 and len(set(identities)) == q, "native16 state requires distinct16 views")
    if not synthetic and q == 400:
        _require(len(set(identities)) == 400, "q400 state requires400 distinct views")
    _require(q >= 4 and q % 4 == 0, "invalid sharing batch horizon")
    _require(type(batch_size) is int and batch_size in ((16,) if native16 else (2, 64) if synthetic else (64,)), "sharing batch size mismatch")
    order = _updates(learner, q, protocol_id)
    _count(completed_updates, "completed updates")
    _require(completed_updates <= len(order), "sharing horizon exceeded")
    _require(len(identities) == q and all(isinstance(v, str) and v for v in identities)
             , "missing ordered views")
    vectors = {"source_valid_per_view": list(source_valid_per_view),
               "target_valid_per_view": list(target_valid_per_view),
               "padded_per_view": list(padded_per_view)}
    for name, values in vectors.items():
        _require(len(values) == q, f"wrong {name} length")
        for value in values:
            _count(value, name)
            _require(value > 0, f"empty {name}")
    _require(all(p >= s for p, s in zip(vectors["padded_per_view"], vectors["source_valid_per_view"])),
             "padded/source count mismatch")
    counts = {site: order[:completed_updates].count(site) for site in SHARING_SITES}
    sharing = {"schema": "sharing-state-v1", "synthetic": synthetic, "q": q,
               "batch_size": batch_size, "learner": learner, "trained_sites": list(SHARING_SITES),
               "horizon": len(order), "ordered_view_identities": identities,
               **vectors, "completed_per_site": counts,
               "source_valid_per_site": {s: sum(vectors["source_valid_per_view"][:n]) for s, n in counts.items()},
               "target_valid_per_site": {s: sum(vectors["target_valid_per_view"][:n]) for s, n in counts.items()},
               "padded_per_site": {s: sum(vectors["padded_per_view"][:n]) for s, n in counts.items()},
               "next_site": order[completed_updates] if completed_updates < len(order) else None}
    if native16 or (not synthetic and q == 400):
        sharing["protocol_id"] = protocol_id
    stream = {"completed_updates": completed_updates, "offset": completed_updates * batch_size,
              "view_offsets": dict(counts), "next_site": sharing["next_site"]}
    return sharing, stream


def _site_policy(policy, *, learner="shared"):
    expected = {site: "trained" for site in SHARING_SITES} if learner == "shared" else {
        learner.removeprefix("specialist_"): "trained"}
    if learner == "shared":
        expected["enc_l14"] = "heldout_after_freeze"
    _require(isinstance(policy, dict) and set(policy) == set(expected), "evaluation sites differ from sharing policy")
    topologies = set()
    for name, role in expected.items():
        declaration = policy[name]
        _require(isinstance(declaration, dict) and set(declaration) == {"site", "role"}
                 and declaration["role"] == role, "invalid sharing site role")
        site = declaration["site"]
        _require(isinstance(site, dict),
                 "evaluation site identity mismatch")
        # The routing resolver verifies full topology against the actual backbone.
        from .models.split_model import ResolvedSite
        try:
            resolved = ResolvedSite(**site)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid resolved sharing site") from error
        _require(resolved.site_id == name and resolved.schema == "encoder-site-v2",
                 "sharing site identity/schema mismatch")
        _require(type(resolved.layer_count) is int and resolved.layer_count > 19
                 and type(resolved.hidden_dim) is int and resolved.hidden_dim > 0
                 and isinstance(resolved.model_revision, str) and bool(resolved.model_revision),
                 "invalid sharing topology")
        _require((name == "enc_fn" and resolved.where == "after_final_norm" and resolved.index is None)
                 or (name != "enc_fn" and resolved.where == "after_layer"
                     and type(resolved.index) is int and resolved.index == int(name[5:])),
                 "invalid sharing boundary")
        expected_path = "encoder.norm" if name == "enc_fn" else f"encoder.layers.{resolved.index}"
        _require(resolved.module_path == expected_path, "sharing module path mismatch")
        topologies.add((resolved.model_revision, resolved.layer_count, resolved.hidden_dim))
    _require(len(topologies) == 1, "sharing sites have inconsistent backbone topology")


def validate_sharing_state(payload):
    """Reject drift in sharing kind, phase, schedule, views and exposure."""
    if payload.get('metadata', {}).get('protocol_identity') == 'encoder-original6-native16-stages-v1':

        return validate_stages_state(payload)
    if payload.get('metadata', {}).get('protocol_identity') == 'encoder-original6-native16-pilot-v1':

        return validate_pilot_state(payload)
    _require(isinstance(payload, dict) and payload.get("schema") == "experiment-state-v2",
             "invalid sharing state envelope")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("sharing"), dict):
        raise ValueError("missing sharing metadata")
    sharing = metadata["sharing"]
    fields = ("synthetic", "learner", "ordered_view_identities", "source_valid_per_view",
              "target_valid_per_view", "padded_per_view", "batch_size")
    _require(all(field in sharing for field in fields), "incomplete sharing metadata")
    count = metadata.get("completed_updates")
    expected, stream = build_sharing_state(completed_updates=count,
        protocol_id=metadata.get("protocol_identity"), **{k: sharing[k] for k in fields})
    _require(canonical_bytes(sharing) == canonical_bytes(expected), "sharing counters/order/horizon mismatch")
    kind = "shared_encoder_v2" if sharing["learner"] == "shared" else "single_site_v2"
    _require(payload.get("kind") == kind, "sharing learner/kind mismatch")
    _require(metadata.get("phase") == "both", "sharing requires simultaneous objective phase")
    _require(canonical_bytes(payload.get("stream")) == canonical_bytes(stream), "sharing stream/view offsets mismatch")
    _require(isinstance(payload.get("scheduler"), dict) and type(payload["scheduler"].get("last_epoch")) is int
             and payload["scheduler"]["last_epoch"] == count,
             "sharing scheduler/global update mismatch")
    _site_policy(metadata.get("evaluation_sites"), learner=sharing["learner"])
    if kind == "single_site_v2":
        site = sharing["learner"].removeprefix("specialist_")
        _require(metadata.get("site") == metadata["evaluation_sites"][site]["site"], "specialist saved site mismatch")


def _freeze_reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_digest(path)}


def _read_reference(reference):
    _require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "invalid freeze file reference")
    path = Path(reference["path"])
    _require(path.is_file() and file_digest(path) == reference["sha256"], "freeze evidence bytes changed")
    return path


def _verified_runs(trained_runs, *, synthetic, protocol_identity, site_policy, recipe_identity, architecture_decision):
    """Recompute completion from retained events and inspect checkpoint bytes."""
    import torch
    from .experiment_state import _validate

    _require(isinstance(trained_runs, dict) and set(trained_runs) == set(LEARNERS), "freeze requires all four learners")
    checkpoint_hashes = []
    common = None
    for learner, run in trained_runs.items():
        _require(isinstance(run, dict) and set(run) == {"root", "manifest", "inventory", "events"}, "invalid trained run references")
        root = Path(run["root"]).resolve()
        paths = {key: _read_reference(run[key]) for key in ("manifest", "inventory", "events")}
        _require(all(path.resolve().is_relative_to(root) for path in paths.values()), "run references escape root")
        manifest = json.loads(paths["manifest"].read_text())
        inventory = json.loads(paths["inventory"].read_text())
        verify_inventory(root, inventory)
        _require(inventory["mode"] == "tensor-complete", "freeze requires retained checkpoint bytes")
        phases = manifest.get("phases")
        _require(isinstance(phases, list) and len(phases) == 1, "freeze requires one simultaneous phase")
        horizon = phases[0].get("updates")
        _count(horizon, "freeze horizon")
        factor = 3 if learner == "shared" else 1
        native16 = protocol_identity in NATIVE16_PROTOCOLS
        long = protocol_identity == NATIVE16_LONG_PROTOCOL
        if native16:
            validate_native16_horizon(horizon // factor)
            _require(synthetic or not long or horizon == 6400 * factor,
                     "formal native16 long freeze requires q6400")
        _require(horizon > 0 and horizon % (4 * factor) == 0
                 and (synthetic or native16 or horizon == (400 if protocol_identity == DEPTH_PROTOCOL else 200) * factor), "freeze horizon mismatch")
        _require(manifest["phases"] == [{"phase_id": "combined", "updates": horizon,
                    "start_step": 0, "start_valid_tokens": 0}], "freeze trained horizon mismatch")
        events = read_events(paths["events"])
        status = completion_status(manifest, events, inventory, root)
        _require(status["status"] == "complete" and status["tensor_bytes_verified"], "trained run incomplete")
        trained_sites = SHARING_SITES if learner == "shared" else (learner.removeprefix("specialist_"),)
        task_steps = [horizon // 2, horizon]
        objective_steps = [horizon * i // 4 for i in range(1, 5)]
        if long:
            task_steps = []
            objective_steps = ([horizon // 3, 2 * horizon // 3, horizon] if learner == "shared"
                               else [horizon // 2, horizon])
        if learner == "shared":
            if not long:
                task_steps.insert(0, 0)
            objective_steps.insert(0, 0)
        expected_panels = {(purpose, site, step) for purpose, points in
                           (("task", task_steps), ("objective", objective_steps))
                           for site in trained_sites for step in points}
        observed_panels = []
        observed_ids = []
        for event in events:
            if event["event_type"] != "observation":
                continue
            observation = event["payload"]
            request = observation["request"]
            _require(observation["status"] == "complete" and request["role"] == "trained"
                     and request["task"] == "hellaswag" and request["conditions"] == (["no_noise"] if long else ["no_noise", -6, 0, 6, 12, 18]),
                     "freeze requires complete trained-site observations")
            observed_panels.append((request["purpose"], request["site"], request["step"]))
            observed_ids.append(observation["identity"])
        _require(set(observed_panels) == expected_panels and len(observed_panels) == len(expected_panels)
                 and set(manifest["expected_observations"]) == set(observed_ids),
                 "freeze trained observation cadence mismatch")
        steps = []
        checkpoints_by_step = {}
        checkpoint_sharing = None
        for artifact in inventory["artifacts"]:
            if artifact["kind"] != "tensor":
                continue
            payload = torch.load(root / artifact["path"], map_location="cpu", weights_only=False)
            _validate(payload)
            validate_sharing_state(payload)
            metadata = payload["metadata"]
            sharing = metadata["sharing"]
            checkpoint_sharing = sharing
            _require(sharing["learner"] == learner and sharing["synthetic"] is synthetic and sharing["horizon"] == horizon,
                     "freeze checkpoint learner mismatch")
            _require(metadata["protocol_identity"] == protocol_identity, "freeze checkpoint protocol mismatch")
            _require(metadata.get("recipe_identity") == recipe_identity
                     and metadata.get("architecture_decision") == architecture_decision,
                     "freeze checkpoint recipe/architecture decision mismatch")
            expected_policy = site_policy if learner == "shared" else {
                learner.removeprefix("specialist_"): site_policy[learner.removeprefix("specialist_")]}
            _require(metadata["evaluation_sites"] == expected_policy, "freeze checkpoint site policy mismatch")
            paired = {key: sharing[key] for key in ("q", "ordered_view_identities", "source_valid_per_view",
                      "target_valid_per_view", "padded_per_view", "batch_size")}
            paired["initialization_identity"] = metadata["initialization_identity"]
            paired["source_identity"] = metadata["source_identity"]
            if common is None:
                common = paired
            _require(paired == common, "freeze learners lack matched initialization/stream/source")
            steps.append(metadata["completed_updates"])
            checkpoints_by_step[metadata["completed_updates"]] = artifact["sha256"]
            checkpoint_hashes.append(artifact["sha256"])
        expected_steps = ([horizon // 3, 2 * horizon // 3, horizon] if learner == "shared"
                          else [horizon // 2, horizon]) if long else [horizon * i // 4 for i in range(1, 5)]
        if learner == "shared":
            expected_steps.insert(0, 0)
        _require(sorted(steps) == expected_steps, "missing/duplicate report checkpoint cadence")
        if checkpoint_sharing is None:
            raise ValueError("freeze requires sharing checkpoints")
        local_counts: dict[str, int] = dict.fromkeys(SHARING_SITES, 0)
        expected_order = _updates(learner, checkpoint_sharing["q"], protocol_identity)
        updates = [event["payload"] for event in events if event["event_type"] == "update"]
        _require(len(updates) == len(expected_order), "freeze update count mismatch")
        for row, site in zip(updates, expected_order):
            local = local_counts[site]
            _require(row["site_id"] == site and row["site_step"] == local + 1
                     and row["sweep"] == local + 1 and row["objective"]["kind"] == "combined",
                     "freeze update site/order/objective mismatch")
            exposure = row["exposure"]
            _require(exposure["sequences"] == checkpoint_sharing["batch_size"]
                     and exposure["source_tokens"] == checkpoint_sharing["source_valid_per_view"][local]
                     and exposure["target_tokens"] == checkpoint_sharing["target_valid_per_view"][local]
                     and exposure["padded_tokens"] == checkpoint_sharing["padded_per_view"][local],
                     "freeze update exposure differs from saved stream")
            local_counts[site] += 1
        for event in events:
            if event["event_type"] == "observation":
                request = event["payload"]["request"]
                _require(request["checkpoint"] == checkpoints_by_step.get(request["step"])
                         and request["protocol"] == protocol_identity,
                         "freeze observation checkpoint/protocol mismatch")
                _require(request["purpose"] != "objective" or request["objective_kind"] == "combined",
                         "freeze requires simultaneous objective observations")
    _require(len(checkpoint_hashes) == len(set(checkpoint_hashes)), "duplicate checkpoint bytes across learners")
    return sorted(checkpoint_hashes)


def create_study_freeze(path, *, protocol_identity, recipe_identity, architecture_decision,
                        site_policy, trained_runs, obligation_plan, synthetic=False):
    """Publish an immutable freeze only after trained-only manifests complete.

Each run supplies a root and hash-bound manifest/inventory/events references.
The separate obligation plan includes the still-pending held-out task, objective
and geometry work; these pending obligations cannot make trained runs circular.
"""
    for value in (protocol_identity, recipe_identity, architecture_decision):
        _require(isinstance(value, str) and bool(value.strip()), "freeze identities must be explicit")
    _require(type(synthetic) is bool, "synthetic freeze must be explicit")
    _site_policy(site_policy)
    _require(isinstance(obligation_plan, dict) and set(obligation_plan) == {"task", "objective", "geometry"},
             "freeze must bind task/objective/geometry obligations")
    panels = []
    for purpose, values in obligation_plan.items():
        optional_geometry = protocol_identity == NATIVE16_LONG_PROTOCOL and purpose == "geometry"
        _require(isinstance(values, list) and (bool(values) or optional_geometry)
                 and all(isinstance(v, str) and v for v in values),
                 "empty freeze obligation plan")
        panels.extend(values)
    _require(len(panels) == len(set(panels)), "duplicate freeze obligations")
    checkpoints = _verified_runs(trained_runs, synthetic=synthetic,
                                protocol_identity=protocol_identity, site_policy=site_policy,
                                recipe_identity=recipe_identity, architecture_decision=architecture_decision)
    receipt = {"schema": "study-freeze-v1", "status": "complete", "synthetic": synthetic,
               "protocol_identity": protocol_identity, "recipe_identity": recipe_identity,
               "architecture_decision": architecture_decision, "site_policy": site_policy,
               "trained_run_inventory": trained_runs, "checkpoint_hashes": checkpoints,
               "observation_plan": panels, "obligation_plan": obligation_plan}
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".freeze-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_bytes(receipt) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _freeze_reference(path)


def verify_study_freeze(reference, *, checkpoint_sha256=None, protocol_identity=None,
                        site_policy=None, obligation=None):
    """Reverify retained evidence before any real held-out work, including capture."""
    receipt = json.loads(_read_reference(reference).read_text())
    _require(set(receipt) == {"schema", "status", "synthetic", "protocol_identity", "recipe_identity",
             "architecture_decision", "site_policy", "trained_run_inventory", "checkpoint_hashes",
             "observation_plan", "obligation_plan"}, "invalid freeze receipt fields")
    _require(receipt["schema"] == "study-freeze-v1" and receipt["status"] == "complete", "incomplete freeze")
    _site_policy(receipt["site_policy"])
    for key in ("protocol_identity", "recipe_identity", "architecture_decision"):
        _require(isinstance(receipt[key], str) and bool(receipt[key].strip()), "missing freeze identity")
    _require(type(receipt["synthetic"]) is bool, "invalid freeze synthetic flag")
    plan = receipt["obligation_plan"]
    _require(isinstance(plan, dict) and set(plan) == {"task", "objective", "geometry"}
             and all(isinstance(v, list)
                     and (bool(v) or (receipt["protocol_identity"] == NATIVE16_LONG_PROTOCOL and k == "geometry"))
                     and all(isinstance(p, str) and p for p in v) for k, v in plan.items()),
             "invalid freeze obligations")
    panels = [panel for values in plan.values() for panel in values]
    _require(len(set(panels)) == len(panels) and set(panels) == set(receipt["observation_plan"]), "freeze obligation mismatch")
    checkpoints = _verified_runs(receipt["trained_run_inventory"], synthetic=receipt["synthetic"],
                    protocol_identity=receipt["protocol_identity"], site_policy=receipt["site_policy"],
                    recipe_identity=receipt["recipe_identity"], architecture_decision=receipt["architecture_decision"])
    _require(checkpoints == receipt["checkpoint_hashes"], "freeze checkpoint inventory mismatch")
    if checkpoint_sha256 is not None:
        _require(checkpoint_sha256 in checkpoints, "checkpoint absent from freeze")
    if protocol_identity is not None:
        _require(protocol_identity == receipt["protocol_identity"], "freeze protocol mismatch")
    if site_policy is not None:
        _require(site_policy == receipt["site_policy"], "freeze site policy mismatch")
    if obligation is not None:
        _require(obligation in receipt["observation_plan"], "undeclared post-freeze obligation")
    return receipt




def _recipe(config: dict) -> dict:
    """Scientific B recipe, excluding only run destination and derived width."""
    codec = copy.deepcopy(config["codec"])
    codec.pop("input_dim", None)
    training = copy.deepcopy(config["training"])
    training.pop("phase_transfer", None)
    return {key: copy.deepcopy(config[key]) for key in
            ("task", "protocol", "seed", "model", "split", "channel", "data", "evaluation")} | {
                "codec": codec, "training": training}


def parent_recipe_sha256(config: dict) -> str:
    return canonical_digest(_recipe(config))


def validate_phase_transfer(checkpoint: str | Path, config: dict) -> dict:
    """Approve a declared local or functional terminal state; never import optimizer state."""
    path = Path(checkpoint).resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = config["training"].get("phase_transfer")
    parent_kind = expected.get("parent_kind", "local") if isinstance(expected, dict) else "local"
    if parent_kind == "functional":
        return _validate_functional_parent(path, state, config, expected)
    if parent_kind != "local":
        raise ValueError("phase transfer parent_kind must be local or functional")
    record = state.get("phase_record")
    if not isinstance(record, dict) or record.get("phase") != "B-local-reconstruction":
        raise ValueError("phase transfer requires a completed B local checkpoint")
    if record.get("terminal_step") != state.get("step"):
        raise ValueError("local checkpoint is not its declared terminal state")
    if record.get("source_sha256") != training_source_digest():
        raise ValueError("local checkpoint source differs from current source")
    expected = config["training"].get("phase_transfer")
    if not isinstance(expected, dict) or not isinstance(expected.get("parent_recipe_sha256"), str) or not isinstance(expected.get("parent_step"), int):
        raise ValueError("phase two requires a declared B parent recipe digest and terminal step")
    if expected["parent_recipe_sha256"] != parent_recipe_sha256(state["config"]) or expected["parent_step"] != state["step"]:
        raise ValueError("declared B parent recipe or terminal step differs from checkpoint")
    for key in ("task", "protocol", "seed", "model", "split", "channel", "data", "evaluation"):
        if state["config"][key] != config[key]:
            raise ValueError(f"phase-two {key} differs from declared B recipe")
    old_codec, new_codec = copy.deepcopy(state["config"]["codec"]), copy.deepcopy(config["codec"])
    old_codec.pop("input_dim", None)
    new_codec.pop("input_dim", None)
    if old_codec != new_codec:
        raise ValueError("phase-two recipe differs in model, task, data or communication design")
    if canonical_digest(state["data_ids"]) != record.get("data_ids_sha256"):
        raise ValueError("local checkpoint data IDs changed")
    if expected.get("cache_identity_sha256") is not None and expected["cache_identity_sha256"] != record.get("cache_identity_sha256"):
        raise ValueError("declared phase parent/cache does not match checkpoint")
    completion_path = path.with_name("completion.json")
    if not completion_path.is_file():
        raise ValueError("B parent has no completion receipt")
    completion = json.loads(completion_path.read_text())
    parent_sha256 = file_digest(path)
    if (completion.get("status") != "complete" or completion.get("checkpoint") != path.name or
            completion.get("checkpoint_sha256") != parent_sha256):
        raise ValueError("B parent is partial or completion receipt does not match checkpoint")
    if config["training"].get("presentation_stream") is None:
        raise ValueError("phase two requires an exact presentation stream")
    stream = config["training"]["presentation_stream"]
    if stream.get("start_presentation", 0) != record["presentations"]:
        raise ValueError("phase-two stream must start after B's exact presentations")
    if stream.get("seed") != state["config"]["training"]["presentation_stream"].get("seed"):
        raise ValueError("phase-two presentation permutation seed differs from B")
    metadata = {"policy": "communication-weights-only-fresh-optimizer-scheduler-scaler-v1",
                "parent_checkpoint": str(path), "parent_checkpoint_sha256": parent_sha256,
                "parent_recipe_sha256": expected["parent_recipe_sha256"],
                "parent_step": state["step"], "parent_presentations": record["presentations"],
                "cache_identity_sha256": record["cache_identity_sha256"],
                "parent_source_sha256": record["source_sha256"],
                "phase_two_presentations": config["training"]["presentation_stream"],
                "optimizer_state_loaded": False, "scheduler_state_loaded": False,
                "scaler_state_loaded": False}
    return {"state": state, "metadata": metadata}


def _validate_functional_parent(path: Path, state: dict, config: dict, expected: dict) -> dict:
    """Accept a real fresh functional terminal receipt, without relabeling it as B."""
    from .config import load_config

    if state.get("schema") == "experiment-state-v2":
        raise ValueError("versioned state requires carry-state lifecycle, not legacy transfer")
    parent = state["config"]
    settings = parent["training"]
    step = state.get("step")
    if (state.get("phase_record") is not None or state.get("phase_transfer") is not None
            or settings.get("phase_transfer")):
        raise ValueError("functional parent must be a fresh functional first stage")
    if (not isinstance(expected.get("parent_recipe_sha256"), str)
            or type(expected.get("parent_step")) is not int
            or expected["parent_recipe_sha256"] != parent_recipe_sha256(parent)
            or expected["parent_step"] != step or step != settings["max_steps"]):
        raise ValueError("declared functional parent recipe or terminal step differs from checkpoint")
    if expected.get("cache_identity_sha256") is not None:
        raise ValueError("functional parent cannot declare a local replay cache identity")
    for key in ("task", "protocol", "seed", "model", "split", "channel", "data", "evaluation"):
        if parent[key] != config[key]:
            raise ValueError(f"phase-two {key} differs from declared functional recipe")
    old_codec, new_codec = copy.deepcopy(parent["codec"]), copy.deepcopy(config["codec"])
    old_codec.pop("input_dim", None)
    new_codec.pop("input_dim", None)
    if old_codec != new_codec:
        raise ValueError("phase-two communication design differs from functional parent")
    stream, next_stream = settings.get("presentation_stream"), config["training"].get("presentation_stream")
    if not isinstance(stream, dict) or not isinstance(next_stream, dict):
        raise ValueError("functional transfer requires exact presentation streams")
    presentations = step * settings["batch_size"] * settings["gradient_accumulation"]
    if (stream.get("policy") != "epoch-permutations-v1" or next_stream.get("policy") != stream["policy"]
            or stream.get("start_presentation", 0) != 0
            or stream.get("total_presentations") != presentations
            or next_stream.get("start_presentation", 0) != presentations
            or next_stream.get("seed") != stream.get("seed")):
        raise ValueError("functional phase-two stream must follow the parent's exact presentations")
    required = ("completion.json", "run.json", "config.yaml", "data_ids.json")
    if any(not path.with_name(name).is_file() for name in required):
        raise ValueError("functional parent lacks completion, source, recipe or data receipt")
    completion = json.loads(path.with_name("completion.json").read_text())
    run = json.loads(path.with_name("run.json").read_text())
    digest = file_digest(path)
    if expected.get("parent_checkpoint_sha256") != digest:
        raise ValueError("declared functional parent checkpoint SHA-256 differs from weights")
    final = completion.get("final_checkpoint", {})
    if (completion.get("status") != "FULL_BUDGET_COMPLETED"
            or completion.get("reason") != "max_steps"
            or completion.get("source_verified") is not True
            or completion.get("presentation_budget_verified") is not True
            or completion.get("step") != step or completion.get("optimizer_updates") != step
            or completion.get("presentations") != presentations
            or final != {"file": path.name, "step": step, "sha256": digest}):
        raise ValueError("functional parent is partial or terminal completion receipt does not match")
    if run.get("training_source_sha256") != training_source_digest():
        raise ValueError("functional parent source differs from current source")
    if run.get("resume_from") is not None or run.get("phase_transfer") is not None:
        raise ValueError("functional parent must start fresh")
    saved_config = path.with_name("config.yaml")
    if (run.get("resolved_config_sha256") != file_digest(saved_config)
            or parent_recipe_sha256(load_config(saved_config)) != parent_recipe_sha256(parent)):
        raise ValueError("functional parent resolved recipe receipt differs from checkpoint")
    if json.loads(path.with_name("data_ids.json").read_text()) != state["data_ids"]:
        raise ValueError("functional parent data IDs differ from receipt")
    metadata = {"policy": "communication-weights-only-fresh-optimizer-scheduler-scaler-v1",
                "parent_kind": "functional", "parent_checkpoint": str(path),
                "parent_checkpoint_sha256": digest,
                "parent_recipe_sha256": expected["parent_recipe_sha256"],
                "parent_step": step, "parent_presentations": presentations,
                "parent_source_sha256": run["training_source_sha256"],
                "parent_data_ids_sha256": canonical_digest(state["data_ids"]),
                "parent_completion_sha256": file_digest(path.with_name("completion.json")),
                "parent_run_sha256": file_digest(path.with_name("run.json")),
                "phase_two_presentations": next_stream,
                "optimizer_state_loaded": False, "scheduler_state_loaded": False,
                "scaler_state_loaded": False}
    return {"state": state, "metadata": metadata}


def panel_id(observation):
    return f'{observation.purpose}:{observation.learner}:{observation.site}:{observation.step}'


def final_panel_id(observation):
    return panel_id(observation) + ':' + str(observation.conditions[0])


def _controls(learner, metadata):
    return {
        'architecture': canonical_digest(learner.model.codec.config),
        'initialization': metadata['initialization_identity'],
        'training_data': metadata['stream_identity'],
        'objective': canonical_digest(objective_settings('combined')),
        'exposure': canonical_digest({**({'execution_partition': metadata['execution_partition']}
                                       if 'execution_partition' in metadata else {}),
                                      'q': learner.schedule.q, 'batch': learner.effective_batch,
                                      'sites': list(TRAINED_SITES) if learner.schedule.name == 'shared'
                                      else [learner.schedule.name.removeprefix('specialist_')]}),
        'schedule': canonical_digest({'horizon': learner.schedule.horizon,
                                      'warmup': max(1, learner.schedule.horizon // 20),
                                      'policy': 'sharing-linear-warmup-cosine-v1'}),
    }




def pilot_source_inventory():
    return current_source_inventory()


def load_saved_pilot(path):
    """Verify the exact source, configuration, immutable inputs and pool prefix."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    raw = {k: v for k, v in manifest.items() if k != 'manifest_identity'}
    if (manifest.get('schema') != 'encoder-original6-prepared-v1'
            or manifest.get('protocol_id') != ENCODER_PILOT_PROTOCOL
            or manifest.get('manifest_identity') != canonical_digest(raw)
            or manifest.get('source_inventory') != pilot_source_inventory()
            or manifest.get('source_identity') != canonical_digest(manifest['source_inventory'])):
        raise ValueError('pilot manifest or execution source identity mismatch')
    root = path.parent
    config = load_config(_reference(root, manifest['config']))
    if canonical_digest(config) != manifest['config_identity']:
        raise ValueError('pilot resolved configuration identity mismatch')
    batches = [read_prepared_batch(ref, root) for ref in manifest['updates']]
    validation = [read_prepared_batch(ref, root) for ref in manifest['validation']]
    plan = EncoderPilotPlan(tuple(batch_view(b) for b in batches), manifest['study_pairing_id'],
                            synthetic=manifest['synthetic_cpu'])
    training, codec = config['training'], config['codec']
    if (config.get('protocol') != ENCODER_PILOT_PROTOCOL or config['task'] != 'hellaswag'
            or config['seed'] != 0 or config['model']['revision'] != manifest['model_revision']
            or config['model'].get('numerical_policy', 'native') != 'native'
            or config['split'] != PILOT_SITE_SPECS['enc_fn']
            or codec['architecture'] != 'direct_affine' or codec['layernorm'] != 'none'
            or codec['n_res_blocks'] != 0 or codec['snr_film'] or codec['dropout'] != 0
            or training['batch_size'] != 16 or training['gradient_accumulation'] != 1
            or training['lr'] != 2e-4 or training['weight_decay'] != NATIVE16_WEIGHT_DECAY
            or training['schedule_steps'] != plan.horizon or training['max_steps'] != plan.horizon
            or training['kl_weight'] != 1 or training['mse_weight'] != .1
            or training['temperature'] != 1 or training['grad_clip'] != 1
            or training.get('patience') is not None or training.get('warmup_ratio') != .05
            or training.get('min_lr_ratio') != 0 or training.get('valid_only_kl') is not True
            or training.get('loss_weights') is not None):
        raise ValueError('pilot architecture, objective, native batch or fixed horizon differs')
    channel = config['channel']
    if (channel['type'] != 'awgn' or not channel['normalize_power'] or not channel['train_noise']
            or channel['train_snr_range'] != [-6.0, 18.0]):
        raise ValueError('pilot channel differs')
    template = json.loads(_reference(root, manifest['task_request']).read_text())
    validate_task_template(template, config, synthetic=manifest['synthetic_cpu'],
                           final=plan.q == 6400, no_noise_only=plan.q == 6400)
    if not manifest['synthetic_cpu']:
        evaluation_ids = manifest['data_ids']['final_validation' if plan.q == 6400 else 'development']
        if plan.q == 6400 and (config['evaluation']['num_samples'] != 10042
                               or config['evaluation']['snrs'] != ['no_noise']
                               or config['evaluation']['vanilla'] is not False):
            raise ValueError('long study requires full10042 no-noise only, without new vanilla')
        rows = [int(i) for batch in batches for i in batch['row_ids']]
        objective = [int(i) for batch in validation for i in batch['row_ids']]
        pool = manifest['data_ids']['train_rows']
        presentations = plan.q * 16
        expected = PresentationSampler(pool, presentations, seed=20260920).actual_ids.tolist()
        if (len(pool) != 39058 or len(set(pool)) != 39058 or rows != expected
                or len(set(rows)) != min(len(pool), presentations)
                or manifest['presentation_stream'] != {
                    'policy': 'epoch-permutations-v1', 'pool_identity': canonical_digest(pool),
                    'seed': 20260920, 'start_presentation': 0, 'total_presentations': presentations}
                or len(objective) != 128 or len(set(objective)) != 128
                or set(objective) & set(pool)
                or set(objective) != set(manifest['data_ids']['objective_rows'])
                or codec['bottleneck_dim'] != 512 or config['model']['dtype'] != 'bfloat16'
                or config['model']['device'] != 'cuda'
                or manifest['model_revision'] != 'dd0a2683227859151b1730ca3a63087df5b5f39b'
                or config['model']['attn_implementation'] != 'sdpa'
                or template['expected_items'] != (10042 if plan.q == 6400 else 256)
                or template['input_ids'] != evaluation_ids['input_ids']
                or template['source_family_ids'] != evaluation_ids['source_family_ids']
                or template['settings'].get('batch_size') != 16
                or template['settings'].get('num_fewshot') != 5
                or template['settings'].get('scoring_policy') != 'fp32-v1'):
            raise ValueError('pilot pool/selection/development or precision contract differs')
    initialization_path = _reference(root, manifest['initialization'])
    initial = torch.load(initialization_path, map_location='cpu', weights_only=True)
    if state_dict_identity(initial) != manifest['initialization']['state_identity']:
        raise ValueError('pilot initialization tensor identity mismatch')
    return manifest, config, plan, batches, validation, template, initial


def saved_pilot_metadata(manifest):
    return {'source_identity': manifest['source_identity'], 'config_identity': manifest['config_identity'],
            'data_identity': canonical_digest(manifest['data_ids']), 'parent_identity': None,
            'initialization_identity': manifest['initialization']['state_identity'],
            'stream_identity': canonical_digest(manifest['updates']), 'protocol_identity': ENCODER_PILOT_PROTOCOL,
            'prepared_identity': manifest['manifest_identity'], 'model_revision': manifest['model_revision'],
            'final_task_identity': manifest['task_request']['sha256'], 'lineage': [], 'phase': 'both'}


def open_saved_pilot_terminal(training_run, metadata, plan):
    root = Path(training_run).resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    freeze_path = root / 'freeze.json'
    if (campaign['status'] != 'training_frozen' or campaign['completed_updates'] != plan.horizon
            or campaign['prepared_identity'] != metadata['prepared_identity']
            or campaign['freeze_reference'] != {'path': str(freeze_path), 'sha256': file_digest(freeze_path)}):
        raise ValueError('pilot training is not completely frozen for this prepared package')
    freeze = json.loads(freeze_path.read_text())
    if (freeze['schema'] != 'encoder-original6-freeze-v1' or freeze['status'] != 'complete'
            or freeze['prepared_identity'] != metadata['prepared_identity']
            or freeze['final_panels'] != [list(panel) for panel in final_panels(plan.q)]
            or freeze['vanilla_after_codec_panels'] != (None if plan.q == 6400 else 6)):
        raise ValueError('pilot frozen final panel plan differs')
    for reference in freeze['evidence'].values():
        path = Path(reference['path']).resolve()
        if not path.is_relative_to(root) or file_digest(path) != reference['sha256']:
            raise ValueError('pilot freeze evidence bytes changed')
    inventory = json.loads((root / 'inventory.json').read_text())
    verify_inventory(root, inventory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if completion_status(manifest, read_events(root / 'metrics.jsonl'), inventory, root)['status'] != 'complete':
        raise ValueError('pilot training records no longer complete')
    reference = CheckpointRef(**freeze['checkpoint'])
    if Path(reference.path).resolve() != root / f'step_{plan.horizon:06d}.pt':
        raise ValueError('pilot terminal checkpoint path differs')
    state = open_state(reference, expected=metadata)
    if state.payload['metadata']['completed_updates'] != plan.horizon:
        raise ValueError('pilot terminal checkpoint count differs')
    return state, campaign['freeze_reference']


ENCODER_STAGES_CONDITIONS = ('no_noise', -6, 0, 6, 18)


def load_saved_stages(path, *, arm, width):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    raw = {k: v for k, v in manifest.items() if k != 'manifest_identity'}
    if (manifest.get('schema') != 'encoder-stages-prepared-v1'
            or manifest.get('protocol_id') != ENCODER_STAGES_PROTOCOL
            or manifest.get('manifest_identity') != canonical_digest(raw)
            or manifest.get('source_inventory') != stages_source_inventory()
            or manifest.get('source_identity') != canonical_digest(manifest['source_inventory'])):
        raise ValueError('stages source or prepared manifest identity differs')
    root = path.parent
    config = load_config(_reference(root, manifest['configs'][str(width)]))
    batches = [read_prepared_batch(ref, root) for ref in manifest['updates']]
    plan = EncoderStagesPlan(tuple(batch_view(batch) for batch in batches), manifest['study_pairing_id'],
                             arm, width=width, synthetic=manifest['synthetic_cpu'])
    template = json.loads(_reference(root, manifest['task_request']).read_text())
    train, codec, channel = config['training'], config['codec'], config['channel']
    if (config['task'] != 'hellaswag' or config['seed'] != 0 or config['split'] != PILOT_SITE_SPECS['enc_fn']
            or codec['architecture'] != 'direct_affine' or codec['layernorm'] != 'none'
            or codec['n_res_blocks'] != 0 or codec['snr_film'] or codec['dropout'] != 0
            or codec['bottleneck_dim'] != width or train['batch_size'] != 16 or train['gradient_accumulation'] != 1
            or train['lr'] != 2e-4 or train['weight_decay'] != NATIVE16_WEIGHT_DECAY
            or train['schedule_steps'] != plan.horizon or train['max_steps'] != plan.horizon
            or train['kl_weight'] != 1 or train['mse_weight'] != .1 or train['temperature'] != 1
            or train['grad_clip'] != 1 or train.get('patience') is not None
            or train['warmup_ratio'] != .05 or train['min_lr_ratio'] != 0 or train['valid_only_kl'] is not True
            or train.get('loss_weights') is not None or channel['type'] != 'awgn'
            or not channel['normalize_power'] or not channel['train_noise']
            or channel['train_snr_range'] != [-6.0, 18.0]):
        raise ValueError('stages native batch, codec, objective or channel recipe differs')
    if not manifest['synthetic_cpu']:
        pool = manifest['data_ids']['train_rows']
        rows = [int(i) for batch in batches for i in batch['row_ids']]
        expected = PresentationSampler(pool, 160000, seed=20260920).actual_ids.tolist()
        if (len(pool) != 39058 or len(set(pool)) != 39058 or rows != expected
                or manifest['presentation_stream'] != {'policy': 'epoch-permutations-v1',
                    'pool_identity': canonical_digest(pool), 'seed': 20260920,
                    'start_presentation': 0, 'total_presentations': 160000}
                or config['model']['revision'] != 'dd0a2683227859151b1730ca3a63087df5b5f39b'
                or config['model']['dtype'] != 'bfloat16' or config['model']['device'] != 'cuda'
                or config['model']['attn_implementation'] != 'sdpa'
                or template['expected_items'] != 10042 or template['conditions'] != list(ENCODER_STAGES_CONDITIONS)
                or template['input_ids'] != manifest['data_ids']['final_validation']['input_ids']
                or template['source_family_ids'] != manifest['data_ids']['final_validation']['source_family_ids']
                or template['settings']['batch_size'] != 16 or template['settings']['num_fewshot'] != 5
                or template['settings']['scoring_policy'] != 'fp32-v1'):
            raise ValueError('stages full pool, precision or terminal evaluation contract differs')
    reference = manifest['initializations'][str(width)]
    initial = torch.load(_reference(root, reference), map_location='cpu', weights_only=True)
    if state_dict_identity(initial) != reference['state_identity']:
        raise ValueError('stages initial codec state differs')
    metadata = {'source_identity': manifest['source_identity'], 'config_identity': canonical_digest(config),
                'data_identity': canonical_digest(manifest['data_ids']), 'parent_identity': None,
                'initialization_identity': reference['state_identity'], 'stream_identity': canonical_digest(manifest['updates']),
                'protocol_identity': ENCODER_STAGES_PROTOCOL, 'prepared_identity': manifest['manifest_identity'],
                'model_revision': config['model']['revision'], 'final_task_identity': manifest['task_request']['sha256'],
                'lineage': [], 'phase': 'phase1'}
    return manifest, config, plan, batches, template, initial, metadata


BASELINE_PROTOCOL_CONDITIONS = ("no_noise", -6, 0, 6, 12, 18)


def baseline_objective_record_request(learner, state, batches, kind):
    """Declare an observation before executing it; actual draws stay in artifacts."""
    items = []
    for batch in batches:
        ids, families = batch.get("row_ids"), batch.get("source_family_ids")
        if (ids is None or families is None) and not learner.synthetic:
            raise ValueError("objective records require actual row/source-family IDs")
        for index in range(len(batch["labels"])):
            synthetic_id = "synthetic:" + batch_identity({k: v[index:index + 1] if torch.is_tensor(v) else v
                                                        for k, v in batch.items()})
            item_id = str(int(ids[index]) if torch.is_tensor(ids) else ids[index]) if ids is not None else synthetic_id
            source_id = str(families[index]) if families is not None else synthetic_id
            items.append({"item_id": item_id, "source_id": source_id})
    metadata = state.payload["metadata"]
    panel = canonical_digest([batch_identity(b) for b in batches])
    request = {**learner.identity, "checkpoint": state.reference.sha256, "panel": panel,
               "site": "enc_fn", "role": "trained", "task": learner.task,
               "protocol": metadata["protocol_identity"], "noise": canonical_digest({"noise": learner.noise_identity, "panel": panel, "purpose": "objective_validation"}),
               "scorer": "fp32-full-vocabulary-K-masked-sequence-R-v1", "layout": panel,
               "precision": str(next(learner.model.base.parameters()).dtype),
               "backend": str(learner.model.sdpa_backend_policy), "conditions": list(BASELINE_PROTOCOL_CONDITIONS),
               "expected_items": len(items), "learner_kind": "initialization" if learner.completed == 0 else "specialist",
               "step": learner.completed, "site_step": learner.completed, "purpose": "objective",
               "objective_kind": kind,
               "comparison": comparison_refs(learner, metadata, kind),
               "details": {"prompt": panel, "native_policy": panel,
                           "draws": canonical_digest({"noise": learner.noise_identity, "panel": panel, "purpose": "objective_validation"}),
                           "outputs": "baseline-objective-v2"}}
    if getattr(learner.model, "numerical_policy", "native") != "native":
        from .models.precision import compute_provenance
        request["precision"] += ":" + learner.model.numerical_policy
        request["details"]["outputs"] = "baseline-objective-v2:" + canonical_digest({
            "numerical_runtime": compute_provenance(learner.model)})
    return request


def baseline_objective_event_payload(learner, state, records, batches, artifact):
    request = baseline_objective_record_request(learner, state, batches, records[0]["objective"]["kind"])
    items = []
    for batch in batches:
        ids, families = batch.get("row_ids"), batch.get("source_family_ids")
        for index in range(len(batch["labels"])):
            fallback = f"synthetic:{batch_identity(batch)}:{len(items)}"
            item_id = str(int(ids[index]) if torch.is_tensor(ids) else ids[index]) if ids is not None else fallback
            source_id = str(families[index]) if families is not None else fallback
            items.append({"item_id": item_id, "source_id": source_id})
    return {"request": request, "identity": observation_identity(request), "status": "complete", "reuse": None,
            "conditions": [{"condition": row["condition"], "requested": len(items), "completed": len(items),
                            "failed": 0, "failure_reason": None, "denominator": len(items), "metrics": {},
                            "items": items, "objectives": row["objective"]["components"]} for row in records]}


def require_saved_admission(value, message):
    if not value:
        raise ValueError(message)


def verify_admission_reference(ref):
    require_saved_admission(
        isinstance(ref, dict)
        and isinstance(ref.get("path"), str)
        and Path(ref["path"]).is_absolute(),
        "absolute immutable receipt reference required",
    )
    digest = ref.get("sha256")
    require_saved_admission(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest)
        and file_digest(Path(ref["path"])) == digest,
        "receipt file checksum mismatch",
    )
    return Path(ref["path"])


def read_admission_reference(ref):
    return json.loads(verify_admission_reference(ref).read_text())


SHARING_QUALIFICATION_SITES = ("enc_l9", "enc_l19", "enc_fn")


SHARING_QUALIFICATION_GUARD_SHA256 = "1f939a9c7c57e4ada3171a94a731754867d4364e43c62ea255ed8adc3f5d8d93"


SHARING_QUALIFICATION_NATIVE16_MODES = ("native16-readiness-v1", "native16-runnability-v1")


SHARING_QUALIFICATION_NATIVE16_PROBE_WORKLOAD = {"physical_updates": 30, "scientific_updates": 0,
    "effective_batch": 16, "accumulation": 1, "sites": list(SHARING_QUALIFICATION_SITES),
    "specialist_guard_updates": 15, "stress_updates": 9, "shared_updates": 6,
    "objective_sequences": 16, "objective_conditions": ["no_noise", -6, 0, 6, 12, 18],
    "task_documents": 16, "task_conditions": ["no_noise", 0],
    "encoder_reuse": False, "production_qualification": False}


SHARING_QUALIFICATION_CONDITIONS = ["no_noise", -6, 0, 6, 12, 18]


SHARING_QUALIFICATION_FINISH_BEFORE_EPOCH = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc).timestamp()


SHARING_QUALIFICATION_NATIVE16_RUNNABILITY_WORKLOAD = {"physical_updates": 6, "scientific_updates": 0,
    "effective_batch": 16, "accumulation": 1, "sites": list(SHARING_QUALIFICATION_SITES),
    "shared_updates": 6, "checkpoint_roundtrips": 1, "task_documents": 0,
    "objective_conditions": [], "encoder_reuse": False, "production_qualification": False}


def require_saved_qualification(value, message):
    if not value:
        raise ValueError(message)


def validate_saved_qualification_contract(path, *, source_root):
    """Read a saved contract against its explicitly retained historical source.

    Guard and runner bindings are historical bytes, never current entrypoints.
    """
    source_root = Path(source_root).resolve()
    path = Path(path).resolve()
    contract = json.loads(path.read_text())
    require_saved_qualification(
        contract.get("schema") == "sharing-no-norm-qualification-v1",
        "qualification contract required",
    )
    _read_ref(contract["prepared"])
    manifest = load_prepared(contract["prepared"]["path"], source_root=source_root)
    config = manifest["resolved_config"]
    native16 = contract.get("probe_mode") in SHARING_QUALIFICATION_NATIVE16_MODES
    if native16:
        require_saved_qualification(contract.get("target_class", "RTX5090") in ("RTX5090", "Spark"), "explicit supported local target required")
    require_saved_qualification(
        not manifest["synthetic_cpu"]
        and manifest["cell"] == "D-N"
        and manifest["selected_bottleneck_dim"] in (512, 1152, 2304)
        and ((manifest.get("protocol_id") == NATIVE16_PROTOCOL and manifest["q"] == 8
              and manifest["batch_size"] == 16 and manifest.get("execution_partition") is None)
             if native16 else manifest.get("execution_partition") == partition_policy(16))
        and config["codec"]["architecture"] == "direct_affine"
        and config["codec"]["layernorm"] == "none",
        "exact D-N width/native16x4 package required",
    )
    require_saved_qualification(
        config["model"].get("device") == "cuda"
        and config["model"].get("dtype") == "bfloat16"
        and config["model"].get("numerical_policy") == "native"
        and config["model"].get("attn_implementation") == "sdpa"
        and config["model"].get("local_files_only") is True,
        "target qualification requires local CUDA/BF16/native/SDPA model",
    )
    require_saved_qualification(
        contract.get("binding") == binding_identity(manifest),
        "exact diagnostic binding mismatch",
    )
    require_saved_qualification(
        contract.get("diagnostic_only") is True
        and contract.get("automatic_retry") is False
        and contract.get("sole_owner") == "task-7",
        "task-7 diagnostic-only/no-retry admission required",
    )
    cap = contract.get("hard_cap_seconds")
    require_saved_qualification(
        type(cap) is int and 20 <= cap <= min(600 if native16 else 1200, manifest["hard_cap_seconds"]),
        "explicit bounded cap required",
    )
    require_saved_qualification(
        type(contract.get("output_byte_cap")) is int
        and contract["output_byte_cap"] > 0,
        "explicit output byte bound required",
    )
    require_saved_qualification(
        Path(contract["allocation_root"]).resolve() == path.parent,
        "contract allocation root mismatch",
    )
    require_saved_qualification(
        isinstance(contract.get("run_id"), str)
        and contract["run_id"]
        and all(c.isalnum() or c in "-_" for c in contract["run_id"]),
        "safe run id required",
    )
    require_saved_qualification(
        file_digest(source_root / "jscc/sharing_qualification_guard.py")
        == SHARING_QUALIFICATION_GUARD_SHA256,
        "reviewed numerical guard bytes changed",
    )
    require_saved_qualification(
        manifest["source_inventory"].get("scripts/sharing_qualification.py")
        == file_digest(
            source_root / "scripts/sharing_qualification.py"
        ),
        "qualification entry must be in prepared source inventory",
    )
    require_saved_qualification(
        manifest["task_template"]["settings"]["batch_size"] == 16
        and manifest["task_template"]["conditions"] == SHARING_QUALIFICATION_CONDITIONS,
        "exact task16/six conditions required",
    )
    require_saved_qualification(
        Path(contract["hf_home"]).is_absolute(), "absolute pinned HF cache required"
    )
    deadline = contract.get("finish_before_epoch")
    require_saved_qualification(
        type(deadline) in (int, float)
        and math.isfinite(deadline)
        and deadline <= SHARING_QUALIFICATION_FINISH_BEFORE_EPOCH,
        "explicit deadline no later than Oct2 18:00 Taipei required",
    )
    _read_ref(contract["expected_panel"])
    return contract, manifest


def saved_qualification_assets(contract, manifest):
    """Hash exact local model/tokenizer/data assets on the actual execution host."""
    root = Path(manifest["resolved_config"]["model"]["name"]).resolve()
    require_saved_qualification(
        root.is_absolute() and root.is_dir(), "local pinned model directory required"
    )
    assets = contract.get("target_assets")
    required = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "hellaswag-train.arrow",
        "hellaswag-validation.arrow",
    }
    require_saved_qualification(
        isinstance(assets, dict) and set(assets) == required,
        "complete pinned model/tokenizer/data assets required",
    )
    for name, ref in assets.items():
        asset = Path(ref["path"])
        require_saved_qualification(
            asset.is_absolute() and file_digest(asset) == ref["sha256"],
            "target asset mismatch: " + name,
        )
        if not name.endswith(".arrow"):
            require_saved_qualification(
                asset.resolve() == (root / name).resolve(),
                "model asset outside exact configured snapshot",
            )
    expected = contract.get("runtime_versions")
    require_saved_qualification(
        isinstance(expected, dict)
        and set(expected) == {"torch", "transformers", "datasets", "lm-eval"},
        "pinned runtime versions required",
    )
    actual = {name: importlib.metadata.version(name) for name in expected}
    require_saved_qualification(actual == expected, "target runtime versions differ")
    return {"assets_identity": canonical_digest(assets), "runtime_versions": actual}


def saved_qualification_admission(contract):
    from .sharing_admission import validate_overnight_admission

    if contract.get("probe_mode") in SHARING_QUALIFICATION_NATIVE16_MODES:
        # Future native16 dispatch needs exact owner approval and resource release,
        # not a new cumulative-accounting schema or an old16x4 budget approval.
        payload = _read_ref(contract["payload_receipt"])
        workload = SHARING_QUALIFICATION_NATIVE16_RUNNABILITY_WORKLOAD if contract["probe_mode"] == "native16-runnability-v1" else SHARING_QUALIFICATION_NATIVE16_PROBE_WORKLOAD
        require_saved_qualification(payload.get("schema") == "overnight-run-payload-approval-v1"
                 and payload.get("status") == "APPROVED_FOR_BOUND_PAYLOAD"
                 and payload.get("template_only") is False
                 and payload.get("diagnostic_only") is True
                 and payload.get("automatic_retry") is False
                 and payload.get("scientific_promotion") is False
                 and payload.get("ledger_owner") == "task-7"
                 and all(payload.get(k) == contract[k] for k in (
                     "run_id", "binding", "prepared", "target_assets", "expected_panel", "proposal", "finish_before_epoch"))
                 and payload.get("cap_device_seconds") == contract["hard_cap_seconds"]
                 and payload.get("workload") == workload,
                 "exact coordinator-approved native16 diagnostic payload required")
        proposal = _read_ref(contract["proposal"])
        require_saved_qualification(proposal.get("probe_mode") == contract["probe_mode"]
                 and proposal.get("target_class", "RTX5090") == contract.get("target_class", "RTX5090")
                 and proposal.get("binding") == contract["binding"]
                 and proposal.get("hard_cap_seconds") == contract["hard_cap_seconds"]
                 and proposal.get("workload") == workload,
                 "native16 immutable probe scope differs")
        handoff = _read_ref(payload["owner_handoff"])
        host = "Spark" if contract.get("target_class") == "Spark" else "5090B"
        require_saved_qualification(handoff.get("status") == "released" and handoff.get("host") == host
                 and handoff.get("run_id") == contract["run_id"]
                 and bool(payload.get("sole_execution_owner"))
                 and handoff.get("to_owner") == payload["sole_execution_owner"],
                 "fresh execution-owner release required")
        return payload
    return validate_overnight_admission(contract, diagnostic_only=True, query_live=False)


def saved_target_receipt(path, contract, manifest):
    receipt = json.loads(
        (Path(contract["allocation_root"]) / "target-preflight.json").read_text()
    )
    require_saved_qualification(
        receipt.get("status") == "cpu_verified"
        and receipt.get("cuda_initialized") is False
        and receipt.get("full_model_constructed") is False
        and receipt.get("contract_sha256") == file_digest(Path(path))
        and receipt.get("binding") == contract["binding"]
        and receipt.get("hostname") == socket.gethostname()
        and receipt.get("python") == str(Path(sys.executable).resolve()),
        "actual target CPU preflight required",
    )
    actual = saved_qualification_assets(contract, manifest)
    require_saved_qualification(
        all(receipt.get(k) == v for k, v in actual.items()),
        "target assets/runtime changed after preflight",
    )


def validate_saved_qualification_result(result, contract):
    if contract.get("probe_mode") == "native16-runnability-v1":
        require_saved_qualification(result.get("status") == "runnability_completed"
                 and result.get("completed_updates") == 6 and result.get("scientific_updates") == 0
                 and result.get("diagnostic_only") is True and result.get("production_qualified") is False
                 and result.get("binding") == contract["binding"]
                 and result.get("per_site_updates") == dict.fromkeys(SHARING_QUALIFICATION_SITES, 2)
                 and result.get("checkpoint", {}).get("exact") is True
                 and result.get("restored_fresh_state") is True
                 and len(result.get("update_timings", [])) == 6,
                 "six-step native16 runnability receipt required")
        return
    native16 = contract.get("probe_mode") == "native16-readiness-v1"
    require_saved_qualification(
        result.get("status") == "diagnostic_completed"
        and result.get("completed_updates") == (30 if native16 else 24)
        and result.get("diagnostic_only") is True
        and result.get("production_qualified") is False
        and result.get("binding") == contract["binding"]
        and set(result.get("sites", {})) == set(SHARING_QUALIFICATION_SITES),
        "complete exact24 diagnostic receipt required",
    )
    for record in result["sites"].values():
        guard = record.get("guard", {})
        objective = record.get("objective", {})
        tasks = record.get("task", [])
        require_saved_qualification(
            record.get("status") == "diagnostic_completed"
            and guard.get("status") == "passed"
            and guard.get("physical_updates") == 5
            and guard.get("checkpoint", {}).get("exact") is True
            and guard.get("restored_second_update_exact") is True
            and len(record.get("stress_timings", [])) == 3
            and objective.get("sequences") == (16 if native16 else 128)
            and len(objective.get("rows", [])) == 6
            and [row.get("condition") for row in objective["rows"]] == SHARING_QUALIFICATION_CONDITIONS
            and all(
                row.get("completed") == (16 if native16 else 128) and row.get("failed") == 0
                for row in objective["rows"]
            )
            and [row.get("condition") for row in tasks] == (["no_noise", 0] if native16 else SHARING_QUALIFICATION_CONDITIONS)
            and all(row.get("candidate_forward_requests") == (64 if native16 else 1024) for row in tasks),
            "all three guards/reloads/full128objective/full256task panels required",
        )

    if native16:
        require_saved_qualification(result.get("shared_updates") == 6 and result.get("shared_per_site_updates") == dict.fromkeys(SHARING_QUALIFICATION_SITES, 2)
                 and result.get("scientific_updates") == 0, "native16 balanced shared diagnostic required")


def verify_saved_qualification_panel(directory, expected):
    for name, key in [
        ("documents", "sample_id"),
        ("prompts", "prompt_id"),
        ("fewshots", "fewshot_id"),
    ]:
        rows = [
            json.loads(line)
            for line in (directory / (name + ".jsonl")).read_text().splitlines()
        ]
        target = expected[name]
        require_saved_qualification(
            len({r[key] for r in rows}) == len(rows)
            and len({r[key] for r in target}) == len(target)
            and {r[key]: r for r in rows} == {r[key]: r for r in target},
            "task panel identity mismatch: " + name,
        )
        if name == "documents":
            require_saved_qualification(
                [r[key] for r in rows] == [r[key] for r in target],
                "task document order mismatch",
            )


def saved_outer_ln_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_saved_outer_ln_plan(plan):
    runs = {r["run_id"]: r for r in plan["runs"]}
    ref, none = plan["reference_13"], plan["all_outer_none_13_with_reuse"]
    assert not plan["execution_authorized"]
    assert len(runs) == len(plan["runs"]) == 22
    assert len(ref) == len(none) == 13 and len(set(ref) & set(none)) == 4
    assert set(ref) | set(none) == set(runs)
    assert len(plan["pairs"]) == 9
    assert len(plan["first_wave"]) == 6 and len(plan["second_wave"]) == 16
    assert len(set(plan["first_wave"] + plan["second_wave"])) == 22
    assert set(plan["first_wave"] + plan["second_wave"]) == set(runs)
    for a, b in zip(ref, none):
        assert runs[a]["route"] == runs[b]["route"]
        assert runs[b]["main_outer_norm"] == "none"
        assert runs[b]["memory_outer_norm"] in (None, "none")
    for pair in plan["pairs"]:
        a, b = runs[pair["reference"]], runs[pair["candidate"]]
        assert a["split"] == b["split"] and a["memory_enabled"] == b["memory_enabled"]
    assert plan["counts"]["coarse_panels"] == 89
    assert plan["counts"]["coarse_candidate_requests"] == 89 * 10042 * 4
    return runs


def resolve_saved_outer_ln_plan(plan, task):
    runs = validate_saved_outer_ln_plan(plan)
    result = {}
    for key in plan["first_wave"] + plan["second_wave"]:
        entry = runs[key]
        c = copy.deepcopy(task)
        c['run'].update(name=key, output_dir='../runs')
        c['split'] = copy.deepcopy(entry['split'])
        c['codec']['layernorm'] = entry['main_outer_norm']
        c['codec']['snr_film'] = False
        c['codec']['memory'] = {'layernorm': entry['memory_outer_norm'] or 'both', 'snr_film': False}
        validate_config(c)
        t, e = c['training'], c['evaluation']
        assert (t['batch_size'], t['gradient_accumulation'], t['max_steps'], t['schedule_steps']) == (128, 1, 5000, 5000)
        assert t['presentation_stream']['total_presentations'] == 640000
        assert t['save_steps'] == [1000, 2500, 5000] and t['eval_every'] == 250
        assert t['patience'] is None and t['streamed_backward'] and t['valid_only_kl'] and not t['low_sync_logging']
        assert e['batch_size'] == 64 and e['scoring_policy'] == 'fp32-v1' and e['num_samples'] == 10042
        assert e['snrs'] == ['no_noise', -6, 6, 18] and not e['vanilla'] and e['mode'] == 'codec_only'
        result[key] = c
    for pair in plan['pairs']:
        a, b = (copy.deepcopy(result[pair[k]]) for k in ['reference', 'candidate'])
        a.pop('run')
        b.pop('run')
        if pair['changed_component'] == 'main_codec_outer_norm':
            a['codec']['layernorm'] = b['codec']['layernorm']
        else:
            a['codec']['memory']['layernorm'] = b['codec']['memory']['layernorm']
        assert a == b, f'unapproved pair difference: {pair["pair_id"]}'
    return result


def prompt_policy_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate_prompt_policy_run(run, expected, config_hash):
    c = json.loads((run / "completion.json").read_text())
    m = json.loads((run / "run.json").read_text())
    from jscc.presentation import training_source_digest

    if not (
        c["status"] == "FULL_BUDGET_COMPLETED"
        and c["step"] == c["optimizer_updates"] == expected
        and c["presentations"] == 640000
        and c["presentation_budget_verified"]
        and c["source_verified"]
    ):
        raise ValueError("Incomplete fixed study budget")
    if (
        prompt_policy_sha(run / "config.yaml") != m["resolved_config_sha256"]
        or m["training_source_sha256"] != training_source_digest()
    ):
        raise ValueError("Run config/source binding mismatch")
    # The resolved run path differs from the prepared recipe; check the original binding saved at launch.
    if (
        json.loads((run / "launch-binding.json").read_text())["prepared_config_sha256"]
        != config_hash
    ):
        raise ValueError("Prepared config mismatch")
    checkpoint = run / f"step_{expected:06d}.pt"
    if (
        c["final_checkpoint"]["file"] != checkpoint.name
        or prompt_policy_sha(checkpoint) != c["final_checkpoint"]["sha256"]
    ):
        raise ValueError("Checkpoint binding mismatch")
    return checkpoint


def bind_baseline_allocation_result(output, receipt, failure=None):
    """Include physical allocation outcome in the same recomputed completion."""
    from datetime import datetime, timezone
    from jscc.experiment_records import (append_event, artifact_ref, completion_status,
                                        measure, read_events, write_json_atomic)
    output = Path(output)
    write_json_atomic(output / "allocation.json", receipt)
    events = read_events(output / "metrics.jsonl") if (output / "metrics.jsonl").exists() else []
    if not events:
        # No successful record production means no claim of completed work.
        status = {"status": "incomplete", "error": str(failure)}
        write_json_atomic(output / "completion.json", status)
        return status
    template = {key: value for key, value in events[-1].items() if key not in {"event_type", "payload"}}
    template["timestamp"] = datetime.now(timezone.utc).isoformat()
    seconds = sum(a["charged_device_seconds"] for a in receipt["allocations"])
    append_event(output / "metrics.jsonl", {**template, "event_type": "cost", "payload": {
        "category": "gpu_allocation", "seconds": measure(seconds), "attribution": "first_use", "site_id": "enc_fn",
        "scope": "aggregate_physical_device_seconds_current_command; prior separately in allocation.json",
        "concurrency": "explicit_ledger", "device_count": sum(a["devices"] for a in receipt["allocations"])}})
    if failure is not None or receipt["terminal_reason"]:
        append_event(output / "metrics.jsonl", {**template, "event_type": "failure", "payload": {
            "reason": str(failure or receipt["terminal_reason"]), "reference": "allocation.json"}})
    manifest = json.loads((output / "manifest.json").read_text())
    if "allocation.json" not in manifest["expected_artifacts"]:
        manifest["expected_artifacts"].append("allocation.json")
    write_json_atomic(output / "manifest.json", manifest)
    inventory = {"schema": "experiment-records-v1", "mode": "tensor-complete", "omitted_tensors": [],
                 "artifacts": [artifact_ref(path, output, "tensor" if path.suffix == ".pt" else "metadata")
                               for path in sorted(output.rglob("*")) if path.is_file()
                               and path.name not in {"inventory.json", "completion.json"}]}
    write_json_atomic(output / "inventory.json", inventory)
    status = completion_status(manifest, read_events(output / "metrics.jsonl"), inventory, output)
    write_json_atomic(output / "completion.json", status)
    return status


def check_reconstruction_plan(paths: dict[str, str]) -> dict:
    configs = {arm: load_config(path) for arm, path in paths.items()}
    budgets = {arm: check_reconstruction_recipe(config) for arm, config in configs.items()}
    a, b, c, d = (configs[key] for key in "ABCD")
    if budgets["A"]["updates"] != budgets["B"]["updates"] + budgets["C"]["updates"]:
        raise ValueError("A updates must equal B local plus C functional updates")
    if budgets["C"] != budgets["D"]:
        raise ValueError("C/D phase-two budgets, batches, schedule or presentation/noise seeds differ")
    if budgets["A"]["presentation_start"] != 0 or budgets["B"]["presentation_start"] != 0:
        raise ValueError("A/B presentation streams must start at zero")
    if budgets["C"]["presentation_start"] != budgets["B"]["presentations"]:
        raise ValueError("C/D must begin after B's exact presentation count")
    if budgets["A"]["presentation_seed"] != budgets["B"]["presentation_seed"] or budgets["A"]["presentation_seed"] != budgets["C"]["presentation_seed"]:
        raise ValueError("A/B/C/D must use the same presentation permutation seed")
    c_training, d_training = copy.deepcopy(c["training"]), copy.deepcopy(d["training"])
    c_training.pop("phase_transfer", None)
    d_training.pop("phase_transfer", None)
    if c_training != d_training:
        raise ValueError("C/D functional training, LR, warmup, objective or selection policies differ")
    if any(a["evaluation"] != other["evaluation"] for other in (b, c, d)):
        raise ValueError("A/B/C/D must use identical task evaluation settings")
    for other in (b, c, d):
        for key in ("task", "protocol", "model", "split", "codec", "channel", "data", "seed"):
            if key == "codec":
                left, right = copy.deepcopy(a[key]), copy.deepcopy(other[key])
                left.pop("input_dim", None)
                right.pop("input_dim", None)
            else:
                left, right = a[key], other[key]
            if left != right:
                raise ValueError(f"A/B/C/D {key} recipes differ")
    declared = c["training"].get("phase_transfer")
    if not isinstance(declared, dict) or declared.get("parent_kind", "local") != "local" or declared.get("parent_recipe_sha256") != parent_recipe_sha256(b) or declared.get("parent_step") != b["training"]["max_steps"]:
        raise ValueError("C declaration does not bind the planned B recipe and terminal step")
    if ("E1" in configs) != ("E" in configs):
        raise ValueError("reset control requires both E1 and E recipes")
    if "E" in configs:
        e1, e = configs["E1"], configs["E"]
        for arm, reference in (("E1", b), ("E", c)):
            candidate = copy.deepcopy(configs[arm])
            reference = copy.deepcopy(reference)
            for recipe in (candidate, reference):
                recipe.pop("run", None)
                recipe["codec"].pop("input_dim", None)
                recipe["training"].pop("phase_transfer", None)
            if candidate != reference:
                raise ValueError(f"{arm} must match {'B first-stage' if arm == 'E1' else 'C phase-two'} recipe")
        declaration = e["training"].get("phase_transfer", {})
        if (declaration.get("parent_kind") != "functional"
                or declaration.get("parent_recipe_sha256") != parent_recipe_sha256(e1)
                or declaration.get("parent_step") != e1["training"]["max_steps"]):
            raise ValueError("E declaration does not bind the planned functional E1 terminal recipe")
        if e1["training"].get("phase_transfer"):
            raise ValueError("E1 must start fresh")
    return {"arms": budgets, "comparison": "A and C match nominal updates; D matches C phase two",
            "cost_accounting": "C inherits B capture and local cost once"}


coco_native_SPLITS = ("enc_emb", "enc_l4", "enc_l9", "enc_l14", "enc_l19", "enc_fn",
          "dec_l0", "dec_l4", "dec_l8", "dec_l12", "dec_l16", "dec_l20", "dec_l24")


coco_native_CONDITIONS = ["no_noise", -6, 6, 18]


coco_native_PRESENTATIONS = 192_000


def coco_native_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def coco_native_positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def coco_native_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def coco_native_selected_indices(receipt):
    indices = receipt.get("selected_indices", list(range(13)))
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index not in range(13) for index in indices)
            or indices != sorted(set(indices)) or 5 not in indices):
        raise ValueError("selected_indices must be sorted unique canonical indices including enc_fn index5")
    return indices


def coco_native_validate_measurements(receipt):
    """Validate the normalized, evidence-bound preflight review contract."""
    coco_native_selected_indices(receipt)
    if receipt.get("version") != "coco-native-measurements-v1" or receipt.get("status") != "PASS":
        raise ValueError("A passed coco-native-measurements-v1 review is required")
    if receipt.get("selection_basis") != "capacity_and_throughput_before_quality":
        raise ValueError("Batch selection must precede quality observations")
    if set(receipt.get("routes", {})) != set(coco_native_SPLITS):
        raise ValueError("Exactly thirteen route correctness receipts are required")
    for split, row in receipt["routes"].items():
        if row.get("status") == "PASS_EXECUTION_WITH_NUMERICAL_LIMITATION":
            limits = row.get("numerical_limitations")
            review = row.get("numerical_review_file")
            if row.get("execution_checks_passed") is not True or not isinstance(limits, list) or not limits or not all(limits):
                raise ValueError(f"{split}: numerical limitations require passed execution checks and a nonempty limitations list")
            if not isinstance(review, str) or not review or review not in receipt.get("evidence_files", {}):
                raise ValueError(f"{split}: numerical limitations require an evidence-bound independent review")
        elif row.get("status") != "PASS":
            raise ValueError("All routes need passed execution checks or reviewed numerical limitations")
        elif row.get("numerical_limitations"):
            raise ValueError(f"{split}: numerical limitations must not be hidden under PASS")
    batch = coco_native_integer(receipt["training_batch_size"], "training batch")
    accumulation = coco_native_integer(receipt["gradient_accumulation"], "accumulation")
    coco_native_integer(receipt["evaluation_batch_size"], "evaluation batch")
    presentations = coco_native_integer(receipt.get("presentations_per_run", coco_native_PRESENTATIONS), "presentations per run")
    if presentations % (batch * accumulation):
        raise ValueError("Effective batch must divide the chosen presentations per run")
    if not receipt.get("evidence_files") or not receipt.get("source_files"):
        raise ValueError("Preflight raw evidence and executed source bindings are required")
    for split, row in receipt["routes"].items():
        if row["timing_representative"] not in coco_native_SPLITS:
            raise ValueError(f"Unknown timing representative for {split}")
        for key in ("seconds_per_update", "generation_seconds_per_image", "selection_seconds"):
            coco_native_positive(row[key], f"{split}.{key}")
        if row.get("timing_kind") not in {"measured", "conservative_proxy"}:
            raise ValueError("Timing must distinguish measurements from proxies")
    for row in receipt["routes"].values():
        representative = receipt["routes"][row["timing_representative"]]
        if representative["timing_kind"] != "measured":
            raise ValueError("Timing proxies must point to an actually measured route")
    coco_native_positive(receipt["vanilla_generation_seconds_per_image"], "vanilla generation")
    return presentations // (batch * accumulation)


def coco_native_numerical_limitations(receipt):
    """Carry raw numerical exceptions into the same later approval card."""
    return [{"split": split, "status": row["status"],
             "numerical_limitations": copy.deepcopy(row["numerical_limitations"]),
             "numerical_review_file": row["numerical_review_file"]}
            for split, row in receipt["routes"].items()
            if row["status"] == "PASS_EXECUTION_WITH_NUMERICAL_LIMITATION"]


def coco_native_verify_bindings(receipt, evidence_root, source_root):
    for key, root in (("evidence_files", evidence_root), ("source_files", source_root)):
        root = Path(root).resolve()
        for relative, expected in receipt[key].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or coco_native_digest(path) != expected:
                raise ValueError(f"Changed or outside-root binding: {key}/{relative}")


def coco_native_estimate(receipt, updates, samples=2000, factor=1.5):
    """Include measured selection and conservative startup/checkpoint allowance.

    Times are planning estimates, not reserved runtime or convergence claims.
    Four-slot lower bound ignores queueing and scheduling differences.
    """
    coco_native_positive(factor, "timing factor")
    if factor < 1:
        raise ValueError("Timing safety factor cannot be less than one")
    rows = []
    for index in coco_native_selected_indices(receipt):
        split = coco_native_SPLITS[index]
        timing = receipt["routes"][split]
        train = factor * (updates * timing["seconds_per_update"] + timing["selection_seconds"]) + 600
        evaluation = factor * len(coco_native_CONDITIONS) * samples * timing["generation_seconds_per_image"] + 600
        rows.append({"split": split, "timing_kind": timing["timing_kind"],
                     "timing_representative": timing["timing_representative"],
                     "train_seconds": train, "evaluation_seconds": evaluation,
                     "train_job_minutes": math.ceil(train / 60),
                     "evaluation_job_minutes": math.ceil(evaluation / 60)})
    vanilla = factor * samples * receipt["vanilla_generation_seconds_per_image"] + 600
    total = sum(row["train_seconds"] + row["evaluation_seconds"] for row in rows) + vanilla
    return {"timing_hardware": receipt.get("hardware", "UNSPECIFIED; no H200 speed claim"),
            "cross_hardware_scaling_applied": False, "safety_factor": factor, "startup_checkpoint_allowance_seconds_per_job": 600,
            "rows": rows, "vanilla_seconds": vanilla, "total_gpu_hours": total / 3600,
            "four_gpu_no_queue_lower_bound_hours": max(total / 4,
                max(row["train_seconds"] + row["evaluation_seconds"] for row in rows)) / 3600,
            "all_jobs_within_48h": all(max(row["train_seconds"], row["evaluation_seconds"]) <= 172800
                                         for row in rows) and vanilla <= 172800,
            "limitations": "Short preflight extrapolation on the recorded hardware only; no WS-to-H200 conversion. Queueing, late-training variability and shared I/O excluded."}


def coco_native_scenarios(receipt):
    """Compare exposure/evaluation choices without selecting one from quality.

    Lower exposures are explicit alternatives, not silently shortened baselines.
    This function also accepts representative-only timing summaries; its caller
    must retain their incomplete correctness coverage and must not dispatch.
    """
    effective = coco_native_integer(receipt["training_batch_size"], "batch") * coco_native_integer(
        receipt["gradient_accumulation"], "accumulation")
    result = []
    for presentations in (64_000, 128_000, 192_000):
        if presentations % effective:
            continue
        for images in (512, 2000):
            updates = presentations // effective
            result.append({"presentations_per_run": presentations,
                           "role": "legacy_nominal_exposure" if presentations == 192_000 else "reduced_exposure_alternative",
                           "effective_batch": effective, "updates": updates,
                           "images_per_panel": images, "quality_equivalence_claimed": False,
                           "execution_authorized": False,
                           "estimates": coco_native_estimate(receipt, updates, images)})
    return result


def coco_native_verify_plan(directory, expected_manifest_hash=None, require_authorized=False):
    """Bind frozen source/config to the approval supplied to each job."""
    directory = Path(directory).resolve()
    manifest_path = directory / "PREPARED-FILES.json"
    if expected_manifest_hash is not None and coco_native_digest(manifest_path) != expected_manifest_hash:
        raise ValueError("Prepared manifest differs from the submitted frozen hash")
    bindings = json.loads(manifest_path.read_text())
    required = {"manifest.json", "APPROVAL-CARD.json", "preflight-review.json", "vanilla.yaml"}
    if not required.issubset(bindings):
        raise ValueError("Prepared manifest omits a required input")
    for relative, expected in bindings.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or coco_native_digest(path) != expected:
            raise ValueError(f"Changed prepared input: {relative}")
    card = json.loads((directory / "APPROVAL-CARD.json").read_text())
    receipt = json.loads((directory / "preflight-review.json").read_text())
    updates = coco_native_validate_measurements(receipt)
    manifest = json.loads((directory / "manifest.json").read_text())
    if tuple(row["experiment"] for row in manifest["runs"]) != coco_native_SPLITS:
        raise ValueError("Incorrect thirteen-route manifest")
    if any(row["config"] not in bindings for row in manifest["runs"]):
        raise ValueError("Prepared manifest omits an experiment config")
    if (card["presentations_per_run"] != receipt.get("presentations_per_run", coco_native_PRESENTATIONS)
            or card["updates"] != updates or card["horizon"] != updates
            or card["fixed_checkpoint"] != f"step_{updates:06d}.pt"):
        raise ValueError("Fixed exposure/checkpoint policy changed")
    indices = coco_native_selected_indices(receipt)
    if (card.get("selected_indices", list(range(13))) != indices or card["runs"] != len(indices)
            or card["panels"] != 4 * len(indices) + 1
            or [row["split"] for row in card["estimates"]["rows"]] != [coco_native_SPLITS[index] for index in indices]):
        raise ValueError("Approved selected scope or timing rows differ from the reviewed receipt")
    limits = coco_native_numerical_limitations(receipt)
    if card.get("numerical_limitations", []) != limits:
        raise ValueError("Approval card must preserve all reviewed numerical limitations")
    if require_authorized:
        if limits and card.get("accept_numerical_limitations") is not True:
            raise ValueError("Formal approval must explicitly accept the recorded numerical limitations")
        if card.get("execution_authorized") is not True or not card.get("approval_reference"):
            raise ValueError("Formal COCO execution is not authorized")
        if card.get("selection_policy", {}).get("locked") is not True:
            raise ValueError("Selection cadence must be approved and locked")
        if card.get("site_four_gpu_cap_verified") is not True:
            raise ValueError("Account-wide four GPU cap must be verified before array execution")
        if not card["estimates"]["all_jobs_within_48h"]:
            raise ValueError("A proposed job exceeds the 48-hour limit")
    return card, receipt, manifest


def coco_native_verify_training_completion(directory, manifest, index, card):
    """COCO has no HellaSwag presentation-stream status; inspect actual fields."""
    import torch
    directory = Path(directory)
    run_text = (directory / manifest["runs"][index]["run_link"]).read_text().strip()
    if not run_text:
        raise ValueError("Completed training link is empty")
    from jscc.recorded_health import resolve_training_run
    entry = manifest["runs"][index]
    logical, run, config = resolve_training_run(run_text, expected_config=directory / entry["config"]
                                              if "config" in entry else None)
    completion = json.loads((run / "completion.json").read_text())
    expected = {"reason": "max_steps", "step": card["updates"],
                "optimizer_updates": card["updates"], "presentations": card["presentations_per_run"]}
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(f"Incomplete COCO training: {key}={completion.get(key)!r}, expected {value!r}")
    checkpoint = run / card["fixed_checkpoint"]
    if checkpoint.resolve().parent != run:
        raise ValueError("Fixed checkpoint outside its training attempt")
    if not checkpoint.is_file():
        raise ValueError("Final fixed-step checkpoint is missing")
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if config is not None and state.get("config") != config:
        raise ValueError("Saved checkpoint configuration differs from bound case")
    if type(state.get("step")) is not int or state["step"] != card["updates"]:
        raise ValueError("Saved checkpoint step differs from fixed training budget")
    return {"run": str(run), "logical_root": str(logical), "completion": completion,
            "checkpoint": str(checkpoint), "checkpoint_sha256": coco_native_digest(checkpoint)}



def validate_saved_baseline_prepared(manifest_path, *, source_root, resource_guard=None):
    """Interpret an exact historical baseline binding against its retained source.

    The explicit source root must contain the matching historical runner bytes;
    current development source is not a reproduction or execution authority.
    No model, data, optimizer or scheduler work is performed.
    """
    from jscc.experiment_schedule import compile_baseline_plan
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "prepared-encfn-baseline-v2":
        raise ValueError("explicit versioned prepared manifest required")
    config_ref = manifest["config"]
    config_path = root / config_ref["path"]
    if file_digest(config_path) != config_ref["sha256"]:
        raise ValueError("prepared config checksum mismatch")
    config = load_config(config_path)
    native_c = manifest.get("native64_c")
    if native_c is not None:
        if native_c != {"schema": "formal-native64-c-v1", "same_shape_guard": True} or resource_guard is None:
            raise ValueError("native64+C requires the approved externally bounded formal wrapper")
        if (config["model"].get("numerical_policy") != "native" or
                config["training"]["batch_size"] != 64 or config["training"]["gradient_accumulation"] != 1):
            raise ValueError("formal C requires actual native64 without accumulation")
    if (config["split"] != {"stack": "enc", "where": "after_final_norm"}
            or config["codec"]["bottleneck_dim"] != 512 or config["seed"] != 0
            or config["training"]["max_steps"] != 400):
        raise ValueError("prepared config differs from enc_fn/B512/seed0/400 protocol")
    training = config["training"]
    if (training["lr"] != 2e-4 or training["weight_decay"] != 0.01 or training["grad_clip"] != 1.0
            or training["batch_size"] * training["gradient_accumulation"] != 64):
        raise ValueError("prepared optimizer/effective batch differs from accepted baseline")
    if (config["codec"]["snr_film"] or config["codec"]["dropout"] != 0
            or config["channel"]["type"] != "awgn" or not config["channel"]["normalize_power"]):
        raise ValueError("baseline requires dropout0/FiLMoff/masked normalized AWGN")
    if config["codec"]["architecture"] == "residual_mlp" and (
            config["codec"]["n_res_blocks"] != 2 or config["codec"]["hidden_dim"] != 1152
            or config["codec"]["activation"] != "gelu"):
        raise ValueError("baseline residual architecture requires two H1152/GELU blocks")
    if config["task"] != "hellaswag":
        raise ValueError("initial scientific baseline is HellaSwag; COCO uses separate protocol")
    plan = compile_baseline_plan(manifest.get("selected_cells", ()), manifest.get("owner_approval"))
    segments = [s for s in plan.segments if s.cell == manifest["cell"] and s.strategy == manifest["strategy"]]
    if len(segments) != 1:
        raise ValueError("segment is not in the explicitly authorized plan")
    segment = segments[0]
    cell_settings = {"D-none": ("direct_affine", "none"), "D-LN": ("direct_outer_ln", "both"),
                     "R-none": ("residual_mlp", "none"), "R-LN": ("residual_mlp", "both")}
    if (config["codec"]["architecture"], config["codec"]["layernorm"]) != cell_settings[segment.cell]:
        raise ValueError("prepared cell topology mismatch")
    # Preparation binds actual executed source bytes, not Git's dirty flag.
    source_root = Path(source_root).resolve()
    inventory = manifest["source_inventory"]
    expected_paths = {str(p.relative_to(source_root)) for p in (source_root / "jscc").rglob("*.py")}
    expected_paths |= {"scripts/encfn_baseline.py", "uv.lock", "pyproject.toml"}
    if set(inventory) != expected_paths:
        raise ValueError("learner source inventory must cover jscc, runner and dependencies")
    for name, digest in inventory.items():
        if file_digest(source_root / name) != digest:
            raise ValueError(f"executed source differs from preparation: {name}")
    metadata = copy.deepcopy(manifest["state_metadata"])
    if metadata["source_identity"] != canonical_digest(inventory):
        raise ValueError("state source identity differs from executed inventory")
    if metadata["config_identity"] != canonical_digest(config):
        raise ValueError("state config identity differs from resolved config")
    if metadata["stream_identity"] != canonical_digest(manifest["updates"]):
        raise ValueError("ordered input stream identity mismatch")
    if len(manifest["updates"]) != 400:
        raise ValueError("complete 400-update stream must be prepared")
    import zipfile
    archive = manifest["source_archive"]
    archive_path = root / archive["path"]
    if file_digest(archive_path) != archive["sha256"]:
        raise ValueError("retained source archive checksum mismatch")
    with zipfile.ZipFile(archive_path) as bundle:
        import hashlib
        for name, digest in inventory.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise ValueError("retained source bytes differ from executed inventory")
    metadata["source_archive"] = {"path": str(archive_path), "sha256": archive["sha256"]}
    if metadata["data_identity"] != canonical_digest(manifest["data_ids"]):
        raise ValueError("state data identity differs from prepared data IDs")
    return manifest, config, metadata
