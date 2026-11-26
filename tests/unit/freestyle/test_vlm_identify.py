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
