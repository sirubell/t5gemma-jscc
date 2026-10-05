"""Saved scientific reader and relocated tool startup regression contracts."""

import jscc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from jscc.activation_replay import file_digest
from scripts.render_shared_terminal_report import evaluation_panels


def test_saved_stages_report_accepts_own_exact_thirty_panels(tmp_path):
    sites = ('enc_emb', 'enc_l4', 'enc_l9', 'enc_l14', 'enc_l19', 'enc_fn')
    conditions = ('no_noise', -6, 0, 6, 18)
    panels = []
    for condition in conditions:
        for site in sites:
            path = tmp_path/f'{site}_{condition}'/'observation.json'
            path.parent.mkdir()
            path.write_text(json.dumps({'status': 'complete',
                'request': {'study_arm': 'R', 'study_width': 512, 'prepared_identity': 'prepared',
                            'checkpoint_sha256': 'own-terminal', 'expected_items': 2},
                'conditions': [{'completed': 2}]}))
            panels.append({'site': site, 'condition': condition, 'checkpoint': 'own-terminal',
                           'observation': {'sha256': file_digest(path)}})
    result = {'status': 'complete', 'arm': 'R', 'width': 512, 'prepared_identity': 'prepared',
              'checkpoint': {'sha256': 'own-terminal'}, 'completed_panels': panels}
    (tmp_path/'result.json').write_text(json.dumps(result))
    assert list(evaluation_panels(tmp_path, expected_arm='R')) == [
        (site, condition) for condition in conditions for site in sites]
    result['completed_panels'][0]['checkpoint'] = 'foreign-terminal'
    (tmp_path/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='own terminal checkpoint'):
        evaluation_panels(tmp_path, expected_arm='R')


def test_relocated_historical_evaluator_starts_directly_without_repo_pythonpath(tmp_path):
    root = Path(jscc.__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, str(root/'scripts/evaluate_historical_weight_screen.py'), '--help'],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'dec_l20' in result.stdout
    assert list(tmp_path.iterdir()) == []
