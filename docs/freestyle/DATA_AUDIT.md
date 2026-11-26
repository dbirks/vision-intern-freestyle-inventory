# Freestyle data audit, triage and splits

Produced by `freestyle/audit.py` (per-image stats + VLM scene triage) and
`freestyle/make_splits.py` (frozen splits). Reproduce with:

```
uv run python -m freestyle.audit          # 69 Collective VLM calls, cached afterwards
uv run python -m freestyle.make_splits    # writes data/splits.json from the frozen table
uv run python -m freestyle.make_splits --check
```

Derived outputs (gitignored): `data/derived/audit/audit.csv`, `audit.json`,
`vlm_cache.json`, `sheet_<scene_label>.jpg` contact sheets. The split file is
`data/splits.json`; **`data/splits.frozen.json` is the reviewed, official
assignment** (identical to `splits.json` today). The random draw in
`make_splits.py` is what produced that table, and it still runs on every
invocation so the two can be compared — but the frozen `groups` / `images` /
`negatives` are copied over it verbatim and any difference is printed as
`frozen_drift` instead of silently changing a split (see *Split rules*).

Both split files sit under `data/`, which the repo rules keep out of git, so the
frozen table is as durable as the machine it was written on — copy it somewhere
private before deleting anything under `data/`.

**Scene labels are model opinions, not ground truth.** They come from
`qwen3-8-flash-next` on the LLM Collective (temperature 0, 1024 px longest side),
corrected only where a human looking at the same frame saw an obvious error
(see *Manual overrides*). Nothing in `data/derived/` is GT; GT lives in `data/gt/`
and does not exist yet.

## Corpus

| | count |
|---|---:|
| photographs audited | 69 |
| from the server (`images/island-cabinet-01/…`) | 54 |
| workstation copies (`images/local/…`) | 15 |
| independent groups (the split unit) | 56 |
| native 1600x1200 | 53 |
| native 2048x1536 | 15 |
| native 640x480 | 1 |

`data/local_images.csv` reports `width=769` for every local file and puts the real
width in `height` (e.g. 769x1600 for a 1600x1200 frame): those columns are wrong.
`freestyle/data.py` therefore measures dimensions from the files and keeps the CSV
values as `csv_width`/`csv_height`. The 54 server rows match their files exactly.

## Scene triage

| scene label | images | meaning |
|---|---:|---|
| `cabinet_full` | 29 | both shelves mostly in frame — the primary material |
| `cabinet_partial` | 11 | one shelf or part of the cabinet |
| `no_cabinet` | 26 | ceiling, desk, room, laptop — nothing to detect |
| `occluded` | 3 | lens covered or pressed against something |
| `unknown` | 0 | answer could not be parsed |

Other fields: `person_visible` in 38 frames (cabinet_full 16, cabinet_partial 9, no_cabinet 12, occluded 1);
motion blur none/mild/severe = 6/48/15; `upside_down` in 3 frames and
`rotated_90` in 1 — two upside-down *cabinet* views (`ab8c7fc6…/043`,
`de19c989…/032`, both added by hand after a visual review), one upside-down
`no_cabinet` bench capture (`d7c48eea…/027`, the VLM's only rotation call) and one
sideways 90° cabinet view (`c40dc230…/019`).

Sanity check against the four frames the raw package's README spot-checked:
`50828e4b…/008` cabinet_full, `bb2c251e…/000` cabinet_partial (mid-swing, severe blur),
`a7f93d3a…/015` no_cabinet (ceiling + laptop + person), `3058e02e…/024` occluded — all four agree.

## Sharpness

Laplacian variance; this camera and JPEG quality make the whole corpus soft.

| quantile | full resolution | downscaled to 512 px |
|---|---:|---:|
| p0 | 0.86 | 4.33 |
| p25 | 7.47 | 50.77 |
| p50 | 14.5 | 71.96 |
| p75 | 20.64 | 142.78 |
| p100 | 75.38 | 228.75 |

The full-resolution column is the usable signal. The 512 px number is kept as a
second feature but it mostly measures resampling: downscaling *raises* the
variance (median 14.5 → 72.0), so it is not comparable across the two columns.

Buckets (`sharp >=20`, `soft 8-20`, `blurry <8`), cut points taken from this corpus:

| scene label | sharp >=20 | soft 8-20 | blurry <8 |
|---|---:|---:|---:|
| `cabinet_full` | 21 | 8 | 0 |
| `cabinet_partial` | 0 | 4 | 7 |
| `no_cabinet` | 3 | 10 | 13 |
| `occluded` | 0 | 0 | 3 |

Only 21 of the 40 candidate (cabinet) frames reach `sharp_full >= 20`; the
best candidate value is 26.1. Luma runs 5.0–146.3 (mean of means 86.0): the cabinet interior is dim and the exposure fights the shelf LEDs.

## Near-duplicates (pHash, Hamming distance <= 8)

32 of 69 images sit in 9 clusters:

| cluster | size | scene labels | splits touched |
|---|---:|---|---|
| c00 | 14 | cabinet_full | train |
| c01 | 3 | no_cabinet | exclude |
| c02 | 3 | no_cabinet | exclude |
| c03 | 2 | cabinet_partial | test |
| c04 | 2 | no_cabinet | exclude |
| c05 | 2 | cabinet_full | train, val |
| c06 | 2 | no_cabinet | exclude |
| c07 | 2 | cabinet_full | train, val |
| c08 | 2 | no_cabinet | exclude |

The 14-frame cluster is the `local/blur` burst plus two server frames at the same
door angle; the two bench captures (`local/bench/*`) duplicate each other. The
size-2 and size-3 clusters are *different openings* whose keyframes look
near-identical — the cabinet did not change between sessions. They are not the
same photograph, but they are nearly the same view, which is the memorisation
risk in *Limitations*.

Clusters are reported, not dropped. `make_splits` writes the pairs that cross a
split boundary into `splits.json -> near_duplicates_across_splits`, and
`--check` prints how many touch `test`:

```
splits ok: 56 groups, 69 images {'train': 22, 'val': 6, 'test': 12, 'exclude': 29},
29 negatives, 0 near-duplicates touching test (<=8 bits), matches frozen
```

Two pairs cross the train/val boundary (`94562f2f…/001` ~ `d622b414…/055`,
distance 2; `ab8c7fc6…/043` ~ `de19c989…/032`, distance 3 — different openings,
near-identical views, and — as it turns out — the same 180° flip). That softens
val slightly but leaks nothing into test.
De-duplicating across groups would drop a quarter of the usable corpus, so the
group unit stays and the pairs are recorded instead.

## Split rules

* Unit = **group**: the event uuid for server frames, `local/blur` for the 12-frame
  burst, one group per other local file. No group straddles two splits;
  `make_splits --check` verifies that on every write.
* Candidate = `cabinet_full` or `cabinet_partial`. `no_cabinet`/`occluded` frames are
  `exclude`d and also listed under `"negatives"` (29 ids) so they stay
  reachable as rejection / false-positive tests. One occluded frame inside an
  otherwise usable group (`bb2c251e…/002`) is excluded while its two siblings stay in `test`.
* Forced: `local/downloads/frame.jpg` → `val` (the one clean, sharp, lit full view —
  the development reference); the `local/blur` burst → `train`.
* `test` = 40% of candidate groups (11 of 28), stratified so it contains at least
  one sharp full view, one partial view, one blurry view, one rotated view, one
  frame containing a person, one 1600x1200 and one 2048x1536 image — every stratum
  was satisfiable and each pick names its rule in the table below. Remaining slots
  come from `random.Random(0)` over the sorted group list.
* `val` takes 20% of candidate groups, `train` the rest.

### The split is frozen

`data/splits.frozen.json` is the reviewed assignment and `data/splits.json` is
regenerated *from it*. `make_splits` still runs the whole generator (audit →
forced placements → stratified picks → seeded draw) and then compares the two:
the frozen `groups`/`images`/`negatives` win, the difference is printed, and
`--check` refuses a file that no longer matches the frozen table. Today the
generator's *draw* no longer lands on the same groups, because the audit labels
moved after the review: marking `ab8c7fc6…/043` and `de19c989…/032`
`upside_down` changed which group satisfies the rotated stratum, and a re-run
would have moved 4 groups (`1281d316…` train→test, `7227dc17…` test→train,
`50828e4b…` test→val, `de19c989…` val→test). That is exactly why the freeze is
authoritative: **no test group changed** (`frozen_drift` on stdout, splits
untouched, byte-for-byte). Changing the split means deleting/renaming the frozen
file, re-running, and reviewing the new table by eye.

### What `test` actually contains

12 images from 11 openings. One row per image that belongs to a `test` group
(the last row is the `occluded` sibling inside `bb2c251e…`, which the negatives
rule keeps out of training and out of scoring):

| image | scene | `sharp_full` | native res | blur | person | rotation | why (group reason) |
|---|---|---:|---|---|---|---|---|
| `3270fb41…/065.jpg` | `cabinet_full` | 26.1 | 1600x1200 | mild | – | – | stratum: sharp full view of both shelves |
| `c40dc230…/019.jpg` | `cabinet_full` | 18.3 | 1600x1200 | mild | yes | 90° | stratum: upside-down / rotated view |
| `e194690f…/002.jpg` | `cabinet_full` | 25.8 | 2048x1536 | mild | – | – | stratum: native 2048x1536 (QXGA) |
| `50828e4b…/008.jpg` | `cabinet_full` | 15.0 | 2048x1536 | mild | – | – | random draw to reach 40% (seed=0) |
| `581c0d5e…/045.jpg` | `cabinet_full` | 20.3 | 2048x1536 | mild | – | – | random draw to reach 40% (seed=0) |
| `733bc6f7…/001.jpg` | `cabinet_partial` | 4.6 | 1600x1200 | severe | yes | – | stratum: partial cabinet view |
| `b5c9000a…/012.jpg` | `cabinet_partial` | 5.7 | 1600x1200 | severe | yes | – | stratum: blurry view |
| `98c44b08…/006.jpg` | `cabinet_partial` | 5.5 | 1600x1200 | severe | yes | – | stratum: frame containing a person |
| `7227dc17…/001.jpg` | `cabinet_partial` | 12.6 | 1600x1200 | severe | yes | – | stratum: native 1600x1200 (UXGA) |
| `10455345…/013.jpg` | `cabinet_partial` | 17.8 | 1600x1200 | mild | yes | – | random draw to reach 40% (seed=0) |
| `bb2c251e…/000.jpg` | `cabinet_partial` | 7.6 | 2048x1536 | severe | – | – | random draw to reach 40% (seed=0) |
| `bb2c251e…/001.jpg` | `cabinet_partial` | 5.3 | 2048x1536 | severe | – | – | random draw to reach 40% (seed=0) |
| `bb2c251e…/002.jpg` | `occluded` | 0.9 | 2048x1536 | severe | – | – | excluded inside this test group |

| axis | test | val | train |
|---|---|---|---|
| scene label | full 5 / partial 7 | full 4 / partial 2 | full 20 / partial 2 |
| sharpness (`>=20` / `8-20` / `<8`) | 3 / 4 / 5 | 3 / 2 / 1 | 15 / 6 / 1 |
| motion blur (`none` / `mild` / `severe`) | 0 / 6 / 6 | 1 / 4 / 1 | 0 / 21 / 1 |
| native resolution (`1600x1200` / `2048x1536`) | 7 / 5 | 5 / 1 | 18 / 4 |
| frames with a person | 6 | 3 | 16 |
| rotated frames | 1 (90°) | 1 (180°) | 1 (180°) |

The test set is deliberately the *hard* slice: 7 of 12 frames show only part of
the cabinet, 6 of 12 are severely smeared, and only 3 reach `sharp_full >= 20`
(`sharp_full` range 4.6–26.1). No test frame is upside-down — the two
upside-down cabinet frames are the train/val pair `ab8c7fc6…/043` and
`de19c989…/032` — so 180° robustness is a train/val concern and a stated gap
(see *Limitations*); the rotated stratum in `test` is the 90° frame.

| split | groups | images | cabinet frames in it |
|---|---:|---:|---:|
| `train` | 11 | 22 | 22 |
| `val` | 6 | 6 | 6 |
| `test` | 11 | 12 | 12 |
| `exclude` | 28 | 29 | 0 |

12 test images from 11 openings: enough to report a number, not enough to trust a
decimal place — which is why the contract asks for group-bootstrap CIs.

### Every group and its reason

| group | src | n | scene labels | max `sharp_full` | res | split | reason |
|---|---|---:|---|---:|---|---|---|
| `10455345-10b3-4baf-bf2e-00bfefc562c7` | server | 1 | cabinet_partial×1 | 17.8 | 1600x1200 | **test** | random draw to reach 40% of candidate groups (seed=0) |
| `3270fb41-c1f0-4a7d-8497-d35d4745123c` | server | 1 | cabinet_full×1 | 26.1 | 1600x1200 | **test** | stratum: sharp full view of both shelves |
| `50828e4b-1e18-43a5-9c51-a53226bcb264` | server | 1 | cabinet_full×1 | 15.0 | 2048x1536 | **test** | random draw to reach 40% of candidate groups (seed=0) |
| `581c0d5e-848f-4292-9dcc-43d0cc6990fd` | server | 1 | cabinet_full×1 | 20.3 | 2048x1536 | **test** | random draw to reach 40% of candidate groups (seed=0) |
| `7227dc17-b4e6-423c-b982-7f6e8b514dd1` | server | 1 | cabinet_partial×1 | 12.6 | 1600x1200 | **test** | stratum: native 1600x1200 (UXGA) |
| `733bc6f7-22da-4ac0-898a-e98ab4ba6738` | server | 1 | cabinet_partial×1 | 4.5 | 1600x1200 | **test** | stratum: partial cabinet view |
| `98c44b08-c6c9-42b3-ab50-89bb71ec9232` | server | 1 | cabinet_partial×1 | 5.5 | 1600x1200 | **test** | stratum: frame containing a person |
| `b5c9000a-8ede-4aba-aa52-edbd897ab0d6` | server | 1 | cabinet_partial×1 | 5.7 | 1600x1200 | **test** | stratum: blurry view |
| `bb2c251e-38ec-4952-a1d2-250436c7dbe5` | server | 3 | cabinet_partial×2, occluded×1 | 7.6 | 2048x1536 | **test** | random draw to reach 40% of candidate groups (seed=0) |
| `c40dc230-fc17-4edc-969a-1fea7043807a` | server | 1 | cabinet_full×1 | 18.3 | 1600x1200 | **test** | stratum: upside-down / rotated view |
| `e194690f-0b7e-4c37-ad88-0fd27481212f` | server | 1 | cabinet_full×1 | 25.8 | 2048x1536 | **test** | stratum: native 2048x1536 (QXGA) |
| `2a1be071-01cc-4564-8d15-38ed23204172` | server | 1 | cabinet_partial×1 | 5.9 | 1600x1200 | **val** | random draw to the val share |
| `5d959b0d-d215-4fad-9458-85a3edfcdd13` | server | 1 | cabinet_full×1 | 12.8 | 1600x1200 | **val** | random draw to the val share |
| `d622b414-bd5d-4152-a256-ccaf8c38d0a5` | server | 1 | cabinet_full×1 | 21.7 | 1600x1200 | **val** | random draw to the val share |
| `d84c1697-b4b0-4122-8793-4ccbf473c8f3` | server | 1 | cabinet_partial×1 | 15.8 | 1600x1200 | **val** | random draw to the val share |
| `de19c989-a538-47d7-9173-a9fce2a49e7e` | server | 1 | cabinet_full×1 | 24.1 | 1600x1200 | **val** | random draw to the val share |
| `local/downloads/frame.jpg` | local | 1 | cabinet_full×1 | 24.9 | 2048x1536 | **val** | forced val: best single view of the cabinet, kept for development |
| `1281d316-d153-4e8e-9db7-fe1ae60974fe` | server | 1 | cabinet_partial×1 | 6.2 | 1600x1200 | **train** | random draw to the train remainder |
| `2ee373c6-d8b6-4837-8c50-8aea807bbe5e` | server | 1 | cabinet_full×1 | 24.6 | 1600x1200 | **train** | random draw to the train remainder |
| `6e91255f-7c6b-43ff-a5b5-04d56cb09573` | server | 1 | cabinet_partial×1 | 16.3 | 2048x1536 | **train** | random draw to the train remainder |
| `94562f2f-62af-40be-aa5d-05bd53a83d77` | server | 1 | cabinet_full×1 | 21.6 | 1600x1200 | **train** | random draw to the train remainder |
| `9718ccea-a049-4d26-9a26-9d96200a1c1c` | server | 1 | cabinet_full×1 | 18.2 | 2048x1536 | **train** | random draw to the train remainder |
| `986ba47c-0312-433b-b2b7-92bb25b8e22a` | server | 1 | cabinet_full×1 | 22.2 | 1600x1200 | **train** | random draw to the train remainder |
| `9fbbf97f-39eb-4d11-9d25-2c50dec09138` | server | 1 | cabinet_full×1 | 21.8 | 2048x1536 | **train** | random draw to the train remainder |
| `ab8c7fc6-943c-4c3d-b95f-3a71f0966ec6` | server | 1 | cabinet_full×1 | 22.1 | 1600x1200 | **train** | random draw to the train remainder |
| `e96172fe-ccb1-420c-9222-c8bf95a9474a` | server | 1 | cabinet_full×1 | 22.7 | 1600x1200 | **train** | random draw to the train remainder |
| `fc592e73-e044-4630-a6d4-8ed58707dd5d` | server | 1 | cabinet_full×1 | 16.8 | 2048x1536 | **train** | random draw to the train remainder |
| `local/blur` | local | 12 | cabinet_full×12 | 21.1 | 1600x1200 | **train** | forced train: 12-frame motion-blur burst kept as a training group |
| `18260cda-bb0d-4e42-8ab6-3c1d6bb702a2` | server | 1 | no_cabinet×1 | 14.5 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `1bcf3a6d-0e54-4696-b609-ff788cc7a5c0` | server | 1 | no_cabinet×1 | 9.5 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `2093ad76-25c3-4bda-a62f-2a847efea938` | server | 1 | no_cabinet×1 | 8.4 | 2048x1536 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `3058e02e-fa33-4d13-ae73-dfcc6ef8c377` | server | 1 | occluded×1 | 2.2 | 2048x1536 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: occluded) |
| `3169fe8b-be61-442e-9b25-1351d9520c5b` | server | 1 | no_cabinet×1 | 7.5 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `395ec9f5-aad8-438a-9fbe-ca65b5df9c75` | server | 1 | no_cabinet×1 | 7.3 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `3e1a1d70-2ba6-4fe8-8591-93ca3fe4b983` | server | 1 | occluded×1 | 4.0 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: occluded) |
| `4399acc1-a961-46b2-87b9-7171208eeb10` | server | 1 | no_cabinet×1 | 3.1 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `54bfc1c1-5624-4c83-a3db-6510af0651fe` | server | 1 | no_cabinet×1 | 75.4 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `568b4d20-dadc-491d-af78-298f8d8113e3` | server | 1 | no_cabinet×1 | 61.6 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `57888e4f-16f0-465a-90fa-05d660602dd4` | server | 1 | no_cabinet×1 | 5.2 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `6417ffce-7913-4f30-8bb2-54333503623c` | server | 1 | no_cabinet×1 | 7.6 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `743aec3a-23f3-4066-9a64-8c29e80549af` | server | 1 | no_cabinet×1 | 7.5 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `77194bb4-488d-486a-bc41-1997ec4ecf01` | server | 1 | no_cabinet×1 | 7.1 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `80666d71-93d0-4d6c-a6da-17a967cb70e1` | server | 1 | no_cabinet×1 | 7.5 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `84ca0487-7f57-42d1-84d4-967265a4c107` | server | 1 | no_cabinet×1 | 7.7 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `a7f93d3a-7b8a-4323-ac0d-fe240bf68c66` | server | 1 | no_cabinet×1 | 10.9 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `aa698640-fdc4-4978-93d4-1f47fc63cc35` | server | 1 | no_cabinet×1 | 10.0 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `ac5a8111-0497-4bc2-a219-475c344bffa1` | server | 1 | no_cabinet×1 | 21.1 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `accf8ad1-68f7-46a2-912c-47b5723f2e51` | server | 1 | no_cabinet×1 | 6.7 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `bd6cd0a5-3001-4f08-996a-cc83c5f452f2` | server | 1 | no_cabinet×1 | 8.1 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `beb63eba-294a-41e4-81ca-07ea02a23291` | server | 1 | no_cabinet×1 | 10.1 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `d7c48eea-9ddb-4be9-bbd2-a90f8e62e74c` | server | 1 | no_cabinet×1 | 10.4 | 2048x1536 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `d95876c7-a418-4548-a891-054cd49a2217` | server | 1 | no_cabinet×1 | 8.6 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `ec7830e2-11cf-4aa4-a382-e09f81821576` | server | 1 | no_cabinet×1 | 7.7 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `ee39ec45-8d4e-4893-998c-29e3eee42852` | server | 1 | no_cabinet×1 | 6.4 | 2048x1536 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `local/bench/size_1.jpg` | local | 1 | no_cabinet×1 | 5.7 | 1600x1200 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |
| `local/bench/vga.jpg` | local | 1 | no_cabinet×1 | 13.0 | 640x480 | **exclude** | no cabinet_full/cabinet_partial frame in group (labels: no_cabinet) |

## Manual overrides

`freestyle/audit_overrides.json` corrects the VLM on 4 images (6 fields).

| image | field | VLM → human |
|---|---|---|
| `local/downloads/frame.jpg` | `motion_blur` | mild → **none** (labels readable; sharpest frame in the corpus) |
| `c40dc230…/019.jpg` | `rotated_90`, `motion_blur` | absent, severe → **true**, **mild** |
| `de19c989…/032.jpg` | `upside_down`, `rotated_90` | false, absent → **true**, **false** |
| `ab8c7fc6…/043.jpg` | `upside_down` | false → **true** |

The last two are the same view from two openings (pHash distance 3), and both
were re-reviewed at full resolution: rotated 180° the two shelf boards become
horizontal, the dark cartridge connector heads point up, and the Coca-Cola shelf
lands on the *lower* board, exactly as in `local/downloads/frame.jpg`; in the
stored orientation the shelf-edge label strip sits above the cartridge row and
the heads hang below the brand colour. `de19c989…/032` had been recorded as
`rotated_90` by an earlier pass — that note described `c40dc230…/019` (whose
shelf line really does run vertically) and is now corrected to `false`.
`rotated_90` is a field we add: the VLM's `upside_down` only covers 180°, and
the two rotations are different failure modes for a detector. Everything else on
the sheets matched the human view, including the awkward `no_cabinet` vs
`occluded` calls on the bench captures.

## Limitations

1. **One cabinet, one arrangement — memorisable.** Every frame shows the same
   physical cabinet and, as far as this audit can tell, the same cartridge
   layout; the cross-opening duplicate clusters are direct evidence. A model can
   score well by memorising *slot positions* instead of reading the boxes. We
   mitigate by scoring per-frame localisation **and** per-box identity (top-1 at
   IoU>=0.5, macro-F1) plus per-class count MAE and exact-inventory rate, so a
   position-memorising model that misidentifies a face still loses points — and we
   report it here as a limitation. There is no second cabinet, no staged
   refill/removal and no arrangement change in this corpus, so **generalisation to
   a different stocking state cannot be measured from these images at all**.
2. **Convenience sample.** 54 of 1,383 declared frames survived retention, and for
   old openings the survivor is whatever the deployment's own frame selector
   flagged as a keyframe — the system under comparison chose the sample. With 29
   non-cabinet frames removed, 40 usable frames remain.
3. **Scene labels are VLM opinions** (one model, temperature 0, downscaled, 4 frames
   hand-corrected over 6 fields). Fit for filtering, not for GT. Cartridge boxes
   and identities are still entirely unlabelled. The two rotation corrections came
   from a human at full resolution *after* the split was frozen, which is why the
   generator and the frozen table now disagree on 4 groups — the labels used to
   stratify are not fixed, the assignments are.
4. **Persons in 38 frames.** Faces and hands are in more than half the corpus, so
   derived crops published anywhere must be re-checked; the images never leave the
   Collective endpoint and our cluster, and `data/` stays gitignored.
5. **Quality envelope.** No candidate frame is genuinely sharp (best
   `sharp_full` 26.1); expect spine text to be unreadable at 1600x1200 and lean on
   colour, box shape and position — exactly what makes limitation 1 bite.
6. **No capture timestamps, no firmware attribution, no per-frame offsets** (raw
   package `GAPS.md`). Frames are orderable only by `frame_index` inside an
   opening, so a mid-swing blur cannot be tied to a door angle.
7. **The `local/` frames are unattributable** — no opening, no timestamp — and the
   best of them is forced into `val`. Development numbers will therefore be read on
   the friendliest view in the corpus; deliberate, but never quote it as the headline.
8. **The audit itself is CPU-only on this host** (~20 s to re-derive all 69 rows
   from the VLM cache, ~20 min for the first run with its 69 model calls); it
   is not the pipeline that will label the corpus at scale (that is a k8s job).
9. **`test` is hard by construction, and small.** The stratifier was told to pull in
   a partial view, a blurry view, a rotated view and a person view, so 7 of 12 test
   frames are `cabinet_partial`, 6 of 12 are severely motion-blurred and only 3
   reach `sharp_full >= 20`, against 15 of 22 in `train`. A model that does well on
   train/val and badly on test is not necessarily broken — the failure modes are
   deliberately over-represented in test; always read test numbers next to the
   per-stratum table above and with group-bootstrap CIs.
10. **Rotation coverage in `test` is 90° only.** The two upside-down cabinet frames
    (`ab8c7fc6…/043` train, `de19c989…/032` val) are a near-identical pair, so 180°
    robustness can be trained and validated but never scored on this split; if
    upside-down captures matter in deployment, that needs new data, not a re-shuffle
    (the frozen split does not move for a label correction).
