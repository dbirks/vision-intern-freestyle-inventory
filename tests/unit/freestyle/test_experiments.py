import pytest

from freestyle.experiments import entry, launch, schema


def config():
    return {'id': 'example', 'family': 'C_det_agnostic_plus_embed',
            'compute': {'target': 'k8s:5090'}, 'localizer': {'type': 'tiled_detector'}}


def test_defaults_and_hash():
    c = schema.resolve(config())
    assert c['data']['label_source'] == 'gt'
    assert c['eval']['score_thr'] == 'val'
    assert schema.content_hash(c, 'abc') == schema.content_hash(config(), 'abc')
    assert schema.content_hash(c, 'abc') != schema.content_hash(c, 'def')
    with pytest.raises(ValueError):
        schema.resolve({**config(), 'family': 'wrong'})


def test_immutable(tmp_path):
    c = schema.resolve(config())
    schema.create_run_dir(c, 'sha', tmp_path)
    with pytest.raises(FileExistsError):
        schema.create_run_dir(c, 'sha', tmp_path)


def test_submit():
    cmd = launch.submit_argv(schema.resolve(config()), 'configs/x.yaml', 'example-abcd', True)
    assert cmd[:3] == ['k8s/submit.sh', 'example-abcd', '5090']
    assert '--dry-run' in cmd and 'rfdetr==1.12.1' in cmd[cmd.index('--pip') + 1]
    assert cmd[cmd.index('--data-root') + 1] == '/work/data'


def test_resume(tmp_path):
    called = []
    stages = [('first', lambda: called.append('first')), ('second', lambda: called.append('second'))]
    entry.run(schema.resolve(config()), tmp_path, tmp_path / 'out', stages)
    entry.run(schema.resolve(config()), tmp_path, tmp_path / 'out', stages)
    assert called == ['first', 'second']
