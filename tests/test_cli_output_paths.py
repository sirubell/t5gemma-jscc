"""Exercise CLI completion links with directories omitted from deployment."""
import sys
import pytest


@pytest.mark.parametrize('phase', ['train', 'evaluate'])
def test_cli_creates_missing_link_parent(tmp_path, monkeypatch, phase):
    link = tmp_path / 'omitted-directory' / 'run.txt'
    output = tmp_path / 'completed-run'
    def completed(*args, **kwargs):
        assert link.parent.is_dir()
        return output

    if phase == 'train':
        import train as cli
        import jscc.training as implementation
        monkeypatch.setattr(cli, 'load_config', lambda _: {})
        monkeypatch.setattr(implementation, 'train', completed)
        argv = ['train.py', '--config', 'dummy.yaml', '--run-path-file', str(link)]
    else:
        import evaluate as cli
        import jscc.evaluation as implementation
        monkeypatch.setattr(implementation, 'evaluate', completed)
        argv = ['evaluate.py', '--run', 'dummy', '--output-path-file', str(link)]
    monkeypatch.setattr(sys, 'argv', argv)
    cli.main()
    assert link.read_text() == str(output) + '\n'
