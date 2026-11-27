"""Tests for the CPU chain-check schema options added to de-risk the GPU window."""
import json
import tempfile
from pathlib import Path

import pytest

from freestyle.experiments import entry, launch, schema


def base():
    return {'id': 'Z0_x', 'family': 'B_zeroshot_plus_embed',
            'compute': {'target': 'k8s:5090', 'mode': 'none'},
            'localizer': {'type': 'zeroshot'}}


def test_compute_mode_defaults_to_auto():
    c = schema.resolve({**base(), 'compute': {'target': 'k8s:5090'}})
    assert c['compute']['mode'] == 'auto'


def test_compute_mode_validated():
    assert schema.resolve(base())['compute']['mode'] == 'none'
    with pytest.raises(ValueError):
        schema.resolve({**base(), 'compute': {'target': 'k8s:5090', 'mode': 'gpu'}})


def test_chain_check_default_and_validation():
    assert schema.resolve(base())['chain_check'] is False
    assert schema.resolve({**base(), 'chain_check': True})['chain_check'] is True
    with pytest.raises(ValueError):
        schema.resolve({**base(), 'chain_check': 'yes'})


def test_submit_passes_mode_after_wait_for():
    # --mode must reach submit.sh, and stay after --wait-for so the pip spec (index 8)
    # that test_experiments already asserts on is not shifted.
    cmd = launch.submit_argv(schema.resolve(base()), 'x.yaml', 'z0-x')
    assert cmd[cmd.index('--mode') + 1] == 'none'
    assert cmd.index('--mode') > cmd.index('--wait-for')
    assert cmd[8] == 'transformers==5.24.1 pycocotools scipy'


def test_prompt_list_serialises_as_separate_tokens():
    # Regression: a list-valued localizer knob (prompts) used to be stringified whole,
    # producing the literal token "['cartridge box']" for an argparse nargs='*' flag.
    c = schema.resolve({**base(), 'localizer': {'type': 'zeroshot', 'backend': 'owlv2',
                                                 'prompts': ['cartridge box'], 'tile': 1024}})
    rendered = []

    def fake_propose(args):
        rendered.extend(args)
        raise SystemExit  # stop before real inference

    import freestyle.propose.run as propose_run
    orig = propose_run.main
    propose_run.main = fake_propose
    try:
        with pytest.raises(SystemExit):
            entry.stage_split(c, _root(), _out(), 'val')
    finally:
        propose_run.main = orig
    assert '--prompts' in rendered
    assert rendered[rendered.index('--prompts') + 1] == 'cartridge box'
    assert "['cartridge box']" not in rendered
    assert rendered[rendered.index('--tile') + 1] == '1024'


def _root():
    # Minimal data root so stage_split reaches the propose() call.
    root = Path(tempfile.mkdtemp())
    manifest = {'categories': [{'id': 1, 'name': 'cartridge'}],
                'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 4, 'height': 4}],
                'annotations': []}
    (root / 'gt').mkdir(exist_ok=True)
    (root / 'gt' / 'val.json').write_text(json.dumps(manifest))
    (root / 'splits.json').write_text(json.dumps({'images': {'a.jpg': 'val'}}))
    (root / 'raw').mkdir()
    return root


def _out():
    return Path(tempfile.mkdtemp())
