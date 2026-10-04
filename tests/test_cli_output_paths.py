"""Exercise CLI completion links with directories omitted from deployment."""
import sys
import json
import pytest


@pytest.mark.parametrize('phase', ['train', 'train_recipe', 'evaluate', 'evaluate_recorded', 'evaluate_recipe'])
def test_cli_creates_missing_link_parent(tmp_path, monkeypatch, phase):
    link = tmp_path / 'omitted-directory' / 'run.txt'
    output = tmp_path / 'completed-run'
    def completed(*args, **kwargs):
        assert link.parent.is_dir()
        if phase == 'evaluate_recipe':
            output.mkdir()
            (output/'results.json').write_text(json.dumps({'status': 'complete', 'counts': {}}))
        return output

    if phase == 'train':
        import train as cli
        import jscc.training as implementation
        monkeypatch.setattr(cli, 'load_config', lambda _: {})
        monkeypatch.setattr(implementation, 'train', completed)
        authored = tmp_path/'ordinary.yaml'
        authored.write_text('task: hellaswag\n')
        argv = ['train.py', '--config', str(authored), '--run-path-file', str(link)]
    elif phase == 'train_recipe':
        import train as cli
        from jscc import recipe_runs
        monkeypatch.setattr(recipe_runs, 'train_recipe', completed)
        monkeypatch.setattr(recipe_runs, 'recipe_run_record', lambda _: {})
        authored = tmp_path/'recipe.yaml'
        authored.write_text('schema: shared-codec-recipe-v1\n')
        argv = ['train.py', '--config', str(authored), '--output', str(output), '--run-path-file', str(link)]
    else:
        import evaluate as cli
        import jscc.evaluation as implementation
        monkeypatch.setattr(implementation, 'evaluate', completed)
        if phase == 'evaluate_recorded':
            from jscc import ordinary_runs
            monkeypatch.setattr(ordinary_runs, 'is_recorded_root', lambda _: True)
            monkeypatch.setattr(ordinary_runs, 'evaluate_run', completed)
        elif phase == 'evaluate_recipe':
            from jscc import recipe_evaluation
            monkeypatch.setattr(recipe_evaluation, 'is_recipe_run', lambda _: True)
            monkeypatch.setattr(recipe_evaluation, 'evaluate_recipe_run', completed)
        argv = ['evaluate.py', '--run', 'dummy', '--output-path-file', str(link)]
    monkeypatch.setattr(sys, 'argv', argv)
    cli.main()
    assert link.read_text() == str(output) + '\n'
