from PIL import Image

from freestyle.vlm_identify import VLMCropIdentifier


def test_parse_and_fallback(monkeypatch):
    ident = VLMCropIdentifier('mock', {'cola': 'Cola'})
    monkeypatch.setattr(ident, '_ask', lambda *a: '```json\n{"class":"cola","confidence":0.8}\n```')
    assert ident.identify(Image.new('RGB', (10, 10)))['class'] == 'cola'
    monkeypatch.setattr(ident, '_ask', lambda *a: '{"class":"invented"}')
    assert ident.identify(Image.new('RGB', (10, 10)))['class'] == 'unknown'


def test_contact_sheet_index_mapping(monkeypatch):
    ident = VLMCropIdentifier('mock', {'cola': 'Cola'})
    monkeypatch.setattr(ident, '_ask', lambda *a: '[{"index":1,"class":"cola","confidence":0.9}]')
    results = ident.contact_sheet([Image.new('RGB', (10, 10))] * 2)
    assert [r['class'] for r in results] == ['unknown', 'cola']


def test_two_stage_part_and_region(monkeypatch):
    rows = [
        {'id': 'cola_1', 'name': 'Coca-Cola Part 1', 'color_hint': 'red', 'shelf_hint': 'bottom'},
        {'id': 'cola_2', 'name': 'Coca-Cola Part 2', 'color_hint': 'red', 'shelf_hint': 'bottom'},
    ]
    ident = VLMCropIdentifier('mock', rows, mode='two_stage', part_region=(0.25, 0.5, 1, 1))
    calls = []

    def ask(image, prompt):
        calls.append((image.size, prompt))
        return ('{"family":"coca cola","confidence":0.9}' if len(calls) == 1
                else '{"part":2,"confidence":0.8}')

    monkeypatch.setattr(ident, '_ask', ask)
    result = ident.identify(Image.new('RGB', (40, 20)), shelf='bottom')
    assert result == {'class': 'cola_2', 'family': 'coca cola', 'part': 2, 'confidence': 0.8}
    assert calls[1][0] == (90, 30)
    assert 'red box, Coca-Cola logo' in calls[0][1]
    assert 'Caller-provided shelf: bottom' in calls[0][1]


def test_two_stage_unreadable_and_single_family(monkeypatch):
    rows = [{'id': 'diet_1', 'name': 'Diet Coke Part 1'},
            {'id': 'diet_2', 'name': 'Diet Coke Part 2'},
            {'id': 'sprite', 'name': 'Sprite'}]
    ident = VLMCropIdentifier('mock', rows, mode='two_stage')
    responses = iter(['{"family":"diet coke","confidence":0.7}',
                      '{"part":"unreadable","confidence":0.2}',
                      '{"family":"sprite","confidence":0.9}'])
    prompts = []

    def ask(image, prompt):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(ident, '_ask', ask)
    assert ident.identify(Image.new('RGB', (10, 10))) == {
        'class': 'unknown', 'family': 'diet coke', 'part': None, 'confidence': 0.7}
    assert ident.identify(Image.new('RGB', (10, 10)))['class'] == 'sprite'
    assert len(prompts) == 3
    assert 'Caller-provided shelf' not in prompts[0]


def test_two_stage_invalid_responses(monkeypatch):
    ident = VLMCropIdentifier('mock', [{'id': 'aa', 'name': 'AA'}], mode='two_stage')
    monkeypatch.setattr(ident, '_ask', lambda *args: '{"family":"invented","confidence":1}')
    assert ident.identify(Image.new('RGB', (10, 10))) == {
        'class': 'unknown', 'family': 'unknown', 'part': None, 'confidence': 0.0}
