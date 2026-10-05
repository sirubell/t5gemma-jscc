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
        import jscc.ordinary_runs as implementation
        monkeypatch.setattr(cli, 'load_config', lambda _: {})
        monkeypatch.setattr(implementation, 'train_task', completed)
        authored = tmp_path/'ordinary.yaml'
        authored.write_text('task: hellaswag\n')
        argv = ['train.py', '--config', str(authored), '--output', str(output), '--run-path-file', str(link)]
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


@pytest.mark.parametrize("recipe", [False, True])
@pytest.mark.parametrize("relation", ["same", "inside", "ancestor", "alias", "dangling", "used"])
def test_training_link_rejects_overlap_or_replay_before_effects(tmp_path, monkeypatch, recipe, relation):
    import train
    output = tmp_path / "new-parent" / "root"
    authored = tmp_path / "config.yaml"
    authored.write_text("schema: shared-codec-recipe-v1\n" if recipe else "task: coco\n")
    link = tmp_path / "receipt" / "run.txt"
    if relation == "same":
        link = output
    elif relation == "inside":
        link = output / "receipt" / "run.txt"
    elif relation == "ancestor":
        link = output.parent
    elif relation == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(output, target_is_directory=True)
        link = alias / "run.txt"
    elif relation == "dangling":
        link.parent.mkdir()
        link.symlink_to(tmp_path / "missing")
    else:
        link.parent.mkdir()
        link.write_text("retained receipt\n")
    monkeypatch.setattr(train, "load_config", lambda *a: pytest.fail("load config before rejection"))
    error = FileExistsError if relation in ("used", "dangling") else ValueError
    with pytest.raises(error):
        train.main(["--config", str(authored), "--output", str(output), "--run-path-file", str(link)])
    assert not output.exists()
    assert not output.parent.exists()


def test_training_missing_output_rejects_before_config_or_link_parent(tmp_path, monkeypatch):
    import train
    monkeypatch.setattr(train, "load_config", lambda *a: pytest.fail("config load"))
    with pytest.raises(SystemExit) as error:
        train.main(["--config", str(tmp_path / "missing.yaml"), "--run-path-file", str(tmp_path / "new" / "link")])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_training_check_is_output_free(tmp_path, monkeypatch):
    import train
    authored = tmp_path / "config.yaml"
    authored.write_text("task: coco\n")
    monkeypatch.setattr(train, "load_config", lambda *a: {"task": "coco"})
    train.main(["--config", str(authored), "--check", "--run-path-file", str(tmp_path / "new" / "link")])
    assert not (tmp_path / "new").exists()


def test_racing_success_link_does_not_overwrite_or_restart(tmp_path, monkeypatch):
    import train
    from jscc import ordinary_runs
    authored = tmp_path / "config.yaml"
    authored.write_text("task: coco\n")
    output, link = tmp_path / "root", tmp_path / "links" / "run.txt"
    calls = []
    def completed(*args, **kwargs):
        calls.append(args)
        output.mkdir()
        link.write_text("racing receipt\n")
        return output
    monkeypatch.setattr(train, "load_config", lambda *a: {})
    monkeypatch.setattr(ordinary_runs, "train_task", completed)
    with pytest.raises(FileExistsError):
        train.main(["--config", str(authored), "--output", str(output), "--run-path-file", str(link)])
    assert len(calls) == 1 and output.is_dir()
    assert link.read_text() == "racing receipt\n"
