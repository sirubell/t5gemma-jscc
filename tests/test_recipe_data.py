"""Offline cached-row equivalence, YAML selection and indexed continuation."""
import copy
import random
from pathlib import Path

from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, T5Gemma2Config
import yaml

from jscc.activation_replay import canonical_digest, file_digest
from jscc.baseline_protocol import batch_identity, read_prepared_batch
from jscc.config import load_config
from jscc.data.hellaswag import recipe_selection_ids
from jscc.data.hellaswag_prompts import PromptBuilder
from jscc.encoder_stages_preparation import _native_batch, bind_recipe_data_cache
from jscc.experiment_recipe import resolve_recipe
from jscc.original13_inputs import load_recipe_cached_data
from jscc.presentation import PresentationSampler
from jscc.recipe_plan import RecipePlan
from jscc.sharing_protocol import batch_view

ROOT = Path(__file__).resolve().parents[1]


def fixture_cache(tmp_path):
    """Real Arrow and fast-tokenizer fixture; no hub calls or weights."""
    raw = Dataset.from_list([
        {'ctx': f'raw {i}', 'activity_label': 'Activity [artifact]',
         'ctx_a': f'Person {i} enters ' + ('long ' * 900 if i % 3 == 0 else 'room ' * (i % 5)),
         'ctx_b': 'tHE Room', 'endings': ['wrong', f'right {i} ' + 'ending ' * (i % 4)],
         'label': '1', 'source_id': str(i // 2)} for i in range(40)])
    arrow = tmp_path / 'train.arrow'
    with ArrowWriter(path=str(arrow)) as writer:
        writer.write_table(raw.data.table)
    assets = tmp_path / 'assets'
    assets.mkdir()
    impl = Tokenizer(WordLevel({'[PAD]': 0, '[EOS]': 1, '[BOS]': 2, '[UNK]': 3,
        'Person': 4, 'enters': 5, 'long': 6, 'room': 7, 'Room': 8, 'right': 9, 'ending': 10,
        'Activity': 11, ':': 12, **{str(i): 13+i for i in range(40)}}, unk_token='[UNK]'))
    impl.pre_tokenizer = Whitespace()
    impl.post_processor = TemplateProcessing(single='[BOS] $A', special_tokens=[('[BOS]', 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=impl, pad_token='[PAD]', bos_token='[BOS]',
                                        eos_token='[EOS]', unk_token='[UNK]')
    tokenizer.save_pretrained(assets)
    model_config = T5Gemma2Config(encoder={'text_config': {'hidden_size': 1152}},
                                   decoder={'hidden_size': 1152, 'bos_token_id': 2, 'pad_token_id': 0})
    model_config.save_pretrained(assets)
    config = load_config(ROOT / 'configs/tasks/sharing_native16_long_b512.yaml')
    config['data'].update(name='fixture/hellaswag', revision='fixed-data', num_validation=4)
    fixed = {'development': {'input_ids': [0, 1], 'source_family_ids': ['fixed0', 'fixed1'],
                             'noise': {'namespace': 'retained-observation', 'seed': 0}},
             'final_validation': {'input_ids': [0, 1, 2]}}
    path = bind_recipe_data_cache(dataset_path=arrow, assets=assets,
        dataset_name=config['data']['name'], dataset_revision=config['data']['revision'],
        model_revision=config['model']['revision'], fixed_data_ids=fixed, output=tmp_path / 'cache.json')
    return raw, tokenizer, model_config, config, path, fixed


def description(config, q=4, **data):
    config = copy.deepcopy(config)
    config['data'].update(data)
    config['training']['presentation_stream']['total_presentations'] = q * 16
    result = {'config': config, 'updates_per_location': q}
    result['identity'] = canonical_digest(result)
    return result


def load(desc, path):
    return load_recipe_cached_data(desc, cache_manifest=path, checksum=file_digest(path))


def assert_batch_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if torch.is_tensor(actual[key]):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key
    assert batch_view(actual) == batch_view(expected)


# Generated at exact accepted foundation 1306a72fe606a9b9045513c6998f82745c545a02.
RETAINED_VIEW_DIGESTS = ['1976451e0991a30bd88d16b3849c90aa228b24edb77fbdf1801d0f963d2fe587', '65b2b58ba15fc153bd92e512dc5ea430c3d6062221fbd8d78e24e5de96c5a16b', '08b91e4b606ea871371dd5ce573235d06a5b1668bb86b329f36021a01b7550fc', '2709234e78b7edbb5069d068d792ace77b3c96a7c8dbc510f652757259d2cc2e']


def test_retained_batches_all_metadata_and_noise_equivalent(tmp_path):
    raw, tokenizer, model_config, config, path, fixed = fixture_cache(tmp_path)
    stream = load(description(config), path)
    ids = recipe_selection_ids(raw, config['data'], fixed)
    builder = PromptBuilder(raw, ids, config['data']['prompt_policy'])
    sampler = PresentationSampler(ids['train_rows'], 64, seed=20260920)
    expected = []
    for index in range(4):
        batch = _native_batch(sampler.actual_ids[index*16:(index+1)*16].tolist(), index,
                              raw, builder, tokenizer, model_config)
        saved = tmp_path / f'retained-{index}.pt'
        torch.save(batch, saved)
        ref = {'path': saved.name, 'sha256': file_digest(saved), 'view_sha256': batch_identity(batch)}
        retained = read_prepared_batch(ref, tmp_path)
        assert_batch_equal(stream[index], retained)
        expected.append(batch_view(retained))
    assert [view.view_sha256 for view in expected] == RETAINED_VIEW_DIGESTS
    assert stream.views == tuple(expected)
    assert stream.data_ids['development'] == fixed['development']
    assert stream.data_ids['final_validation'] == fixed['final_validation']
    assert all(not set(demos) & set(ids['validation_rows']) for demos in stream[0]['demo_ids'])
    assert any('padding' in roles for roles in stream[0]['token_roles'])
    assert any(view['native_view']['source_length'] > 2048 for view in stream[0]['source_views'])
    assert torch.equal(stream[0]['decoder_attention_mask'], stream[0]['labels'].ne(-100))


def test_seek_observation_rng_and_mutation_isolation(tmp_path):
    _, _, _, config, path, _ = fixture_cache(tmp_path)
    stream = load(description(config), path)
    before = copy.deepcopy(stream[2])
    py_rng, torch_rng = random.getstate(), torch.get_rng_state()
    stream[0]
    stream[3]
    assert stream.position(7, 3)['view_index'] == 2
    assert stream.position(7, 3)['location_index'] == 1
    assert stream.position(12, 3)['complete']
    assert_batch_equal(stream[2], before)
    assert random.getstate() == py_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    changed = stream[2]
    changed['input_ids'].zero_()
    changed['demo_ids'][0].clear()
    changed['source_views'][0]['native_view']['demo_ids'].clear()
    changed['prompt_policy'].clear()
    stream.data_ids['development'].clear()
    assert_batch_equal(stream[2], before)
    for index in (-1, 4, True, 1.5):
        with pytest.raises(IndexError):
            stream[index]
    with pytest.raises(ValueError, match='continuation'):
        stream.position(13, 3)


def test_yaml_subset_budget_seed_and_irrelevant_settings(tmp_path):
    _, _, _, config, path, fixed = fixture_cache(tmp_path)
    task = tmp_path / 'task.yaml'
    task.write_text(yaml.safe_dump(config))
    recipe = yaml.safe_load((ROOT / 'configs/recipes/changed.yaml').read_text())
    recipe['task_config'] = str(task)
    recipe['locations'] = ['enc_emb', 'enc_fn']
    recipe['observation']['sites'] = ['enc_emb']
    recipe['updates_per_location'] = 24
    recipe['overrides']['data'] = {'num_train': 12, 'selection_seed': 23}
    recipe_path = tmp_path / 'recipe.yaml'
    recipe_path.write_text(yaml.safe_dump(recipe))
    resolved = resolve_recipe(recipe_path, model_config_path=path.parent / 'assets/config.json')
    first = load(resolved['trajectories'][0], path)
    recipe['updates_per_location'] = 40
    recipe['locations'].reverse()
    recipe['widths'] = [64]
    recipe['overrides']['training']['lr'] = .002
    recipe_path.write_text(yaml.safe_dump(recipe))
    second = load(resolve_recipe(recipe_path, model_config_path=path.parent / 'assets/config.json')['trajectories'][0], path)
    assert len(first) == 24 and len(second) == 40
    assert first.compatibility_identity == second.compatibility_identity
    assert first.identity != second.identity
    assert_batch_equal(first[0], second[0])
    assert first.data_ids['development'] == fixed['development']
    recipe['overrides']['data'].update(num_train=20, selection_seed=2)
    recipe_path.write_text(yaml.safe_dump(recipe))
    third = load(resolve_recipe(recipe_path, model_config_path=path.parent / 'assets/config.json')['trajectories'][0], path)
    assert first.compatibility_identity != third.compatibility_identity
    assert first.data_ids['train_rows'] != third.data_ids['train_rows']
    assert third.data_ids['development'] == fixed['development']
    changed_seed = description(config)
    changed_seed['config']['training']['presentation_stream']['seed'] = 3
    changed_seed.pop('identity')
    changed_seed['identity'] = canonical_digest(changed_seed)
    fourth = load(changed_seed, path)
    assert fourth.compatibility_identity != load(description(config), path).compatibility_identity


def test_explicit_rows_and_incompatible_inputs(tmp_path):
    raw, _, _, config, path, fixed = fixture_cache(tmp_path)
    selection = recipe_selection_ids(raw, config['data'], fixed)
    rows = selection['train_rows'][:12]
    good = load(description(config, row_ids=rows), path)
    assert good.data_ids['train_rows'] == rows
    for bad in ([], [rows[0]]*8, rows + [selection['validation_rows'][0]], rows + [40]):
        with pytest.raises(ValueError, match='explicit recipe rows'):
            load(description(config, row_ids=bad), path)
    with pytest.raises(ValueError, match='explicit recipe rows'):
        load(description(config, row_ids=rows, num_train=8), path)
    with pytest.raises(ValueError, match='revisions differ'):
        load(description(config, revision='other-revision'), path)
    with pytest.raises(ValueError, match='trajectory identity'):
        load({**description(config), 'identity': 'bad'}, path)


def test_missing_assets_no_download_and_no_full_cache_rehash(tmp_path, monkeypatch):
    _, _, _, config, path, _ = fixture_cache(tmp_path)
    calls = []
    real = file_digest
    monkeypatch.setattr('jscc.original13_inputs.file_digest', lambda p: calls.append(Path(p).name) or real(p))
    stream = load(description(config), path)
    assert len(stream) == 4
    assert calls == ['cache.json']
    (tmp_path / 'train.arrow').unlink()
    with pytest.raises(FileNotFoundError, match='cached asset missing'):
        load(description(config), path)
    with pytest.raises(FileNotFoundError, match='receipt missing'):
        load_recipe_cached_data(description(config), cache_manifest=tmp_path / 'missing', checksum='bad')
    with pytest.raises(ValueError, match='checksum'):
        load_recipe_cached_data(description(config), cache_manifest=path, checksum='bad')


def test_recipe_plan_noise_uses_identical_views(tmp_path):
    _, _, _, config, path, _ = fixture_cache(tmp_path)
    stream = load(description(config), path)
    desc = description(config)
    desc.update(locations=['enc_emb', 'dec_l0'], width=32, ratio=50,
                counts={'optimizer_updates': 8},
                phases=[{'start': 0, 'stop': 8, 'objective_kind': 'combined'}])
    desc.pop('identity')
    desc['identity'] = canonical_digest(desc)
    plan = RecipePlan(desc, stream.views, 'retained-pairing')
    for update in range(8):
        actual = plan.update_at(update)
        assert actual.view == batch_view(stream[update // 2])
        assert actual.noise_key.sequence_view_identity == actual.view.view_sha256


def test_tokenization_once_per_selected_row_not_per_update(tmp_path, monkeypatch):
    import jscc.encoder_stages_preparation as preparation
    _, _, _, config, path, _ = fixture_cache(tmp_path)
    original = preparation._native_rows
    seen = []
    def record(rows, *args):
        seen.extend(rows)
        return original(rows, *args)
    monkeypatch.setattr(preparation, '_native_rows', record)
    stream = load(description(config, q=40), path)
    assert seen == stream.data_ids['train_rows']
    count = len(stream.data_ids['train_rows'])
    assert len(seen) == len(set(seen)) == count
    stream.views
    for index in (0, 39, 0, 20):
        stream[index]
    assert len(seen) == count
    for subset in (-1, 0, True):
        with pytest.raises(ValueError, match='recipe selection'):
            load(description(config, num_train=subset), path)


def test_retained_membership_and_family_exclusions_before_subset_limit(tmp_path):
    raw, _, _, config, _, fixed = fixture_cache(tmp_path)
    base = recipe_selection_ids(raw, config['data'], fixed)
    retained = {**fixed, **base}
    retained['development']['source_family_ids'] = [raw[base['train_rows'][0]]['source_id']]
    # Remove that fixed development family from the acquired eligible pool.
    retained['train_rows'] = [row for row in base['train_rows']
        if raw[row]['source_id'] not in retained['development']['source_family_ids']]
    for seed in (0, 23):
        data = {**config['data'], 'selection_seed': seed, 'num_train': 12}
        ids = recipe_selection_ids(raw, data, retained)
        assert ids['validation_rows'] == base['validation_rows']
        assert len(ids['train_rows']) == 12
        assert set(ids['train_rows']) <= set(retained['train_rows'])
        forbidden = {raw[row]['source_id'] for row in ids['validation_rows']}
        forbidden.update(retained['development']['source_family_ids'])
        assert not {raw[row]['source_id'] for row in ids['train_rows']} & forbidden
    excluded_family_row = base['train_rows'][0]
    with pytest.raises(ValueError, match='explicit recipe rows'):
        recipe_selection_ids(raw, {**config['data'], 'row_ids': [excluded_family_row]}, retained)
    with pytest.raises(ValueError, match='retained recipe training pool'):
        recipe_selection_ids(raw, config['data'], {**retained, 'train_rows': base['train_rows']})
