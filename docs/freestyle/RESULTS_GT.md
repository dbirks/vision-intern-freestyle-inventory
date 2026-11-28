# Human ground-truth results

Counting is scored against quiz counts; per-class counts come from answered crops.
AP50 compares with human-filtered boxes that originate from model proposals; it is not independent localization GT.
Identity accuracy is conditional on a matched human-answered, known-identity crop.
Class inventory and AP exclude incomplete images and count mismatches.
Frames answered "not a usable cabinet view" are excluded from every metric (reason not_usable); they are never zero-cartridge frames.
Thresholds minimize validation total count error, then class count error on usable val frames only; embedding validation uses held-out groups.
When no usable val frame has a human count, each run keeps its own val-selected threshold from runs/<exp>/metrics.json (threshold_source=run_metrics).
The validation split needs human count answers only; unanswered validation crops just shrink the identity gallery.
Training-gallery confidence thresholds are selected on validation; +val galleries hold out the query group during validation and use all validation groups at test time. Train and validation may contain near duplicates.
Sharp-full means cabinet_full with no or mild motion blur; partial/blurry means cabinet_partial or severe blur. Other audit cases are omitted from these subsets.

Excluded images:
- val: island-cabinet-01/de19c989-a538-47d7-9173-a9fce2a49e7e/032.jpg (not_usable), island-cabinet-01/d622b414-bd5d-4152-a256-ccaf8c38d0a5/055.jpg (not_usable), island-cabinet-01/5d959b0d-d215-4fad-9458-85a3edfcdd13/001.jpg (not_usable), island-cabinet-01/d84c1697-b4b0-4122-8793-4ccbf473c8f3/032.jpg (not_usable), island-cabinet-01/2a1be071-01cc-4564-8d15-38ed23204172/005.jpg (not_usable)
- test: island-cabinet-01/c40dc230-fc17-4edc-969a-1fea7043807a/019.jpg (not_usable), island-cabinet-01/50828e4b-1e18-43a5-9c51-a53226bcb264/008.jpg (not_usable), island-cabinet-01/10455345-10b3-4baf-bf2e-00bfefc562c7/013.jpg (not_usable), island-cabinet-01/7227dc17-b4e6-423c-b982-7f6e8b514dd1/001.jpg (not_usable), island-cabinet-01/733bc6f7-22da-4ac0-898a-e98ab4ba6738/001.jpg (not_usable), island-cabinet-01/98c44b08-c6c9-42b3-ab50-89bb71ec9232/006.jpg (not_usable), island-cabinet-01/b5c9000a-8ede-4aba-aa52-edbd897ab0d6/012.jpg (not_usable), island-cabinet-01/bb2c251e-38ec-4952-a1d2-250436c7dbe5/000.jpg (not_usable), island-cabinet-01/bb2c251e-38ec-4952-a1d2-250436c7dbe5/001.jpg (not_usable)
- not usable (uncountable frames, never counted as zero): 9 test, 5 val

| Localiser | Identifier | Pair | Thr source | Score thr | ID thr | Train confidence | Train crops | Subset | Images | Total MAE (95% CI) | Top MAE | Bottom MAE | Exact inventory | ID accuracy | AP50 |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| B1_owlv2_dinov2-4c99011d | none | off | val | 0.050 | 0.000 | — | 0 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | — | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | none | off | val | 0.050 | 0.000 | — | 0 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | — | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | none | off | val | 0.050 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | none | on | val | 0.050 | 0.000 | — | 0 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | — | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | none | on | val | 0.050 | 0.000 | — | 0 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | — | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | none | on | val | 0.050 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | vlm | off | val | 0.050 | 0.250 | — | 0 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.500, 0.727] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm | off | val | 0.050 | 0.250 | — | 0 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.500, 0.727] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm | off | val | 0.050 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | vlm | on | val | 0.050 | 0.250 | — | 0 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.500, 0.727] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm | on | val | 0.050 | 0.250 | — | 0 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.500, 0.727] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm | on | val | 0.050 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | val | 0.050 | 0.250 | — | 0 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.640 [0.571, 0.727] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | val | 0.050 | 0.250 | — | 0 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.640 [0.571, 0.727] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | val | 0.050 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | val | 0.050 | 0.250 | — | 0 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.640 [0.571, 0.727] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | val | 0.050 | 0.250 | — | 0 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.640 [0.571, 0.727] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | val | 0.050 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.000 [1.000, 8.000] | 6.000 [1.000, 14.000] | 8.000 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.520 [0.429, 0.636] | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | none | off | val | 0.100 | 0.000 | — | 0 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | — | — | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | none | off | val | 0.100 | 0.000 | — | 0 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | — | — | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | none | off | val | 0.100 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | none | on | val | 0.100 | 0.000 | — | 0 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | — | — | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | none | on | val | 0.100 | 0.000 | — | 0 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | — | — | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | none | on | val | 0.100 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | vlm | off | val | 0.100 | 0.250 | — | 0 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.625 [0.538, 0.727] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm | off | val | 0.100 | 0.250 | — | 0 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.625 [0.538, 0.727] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm | off | val | 0.100 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | vlm | on | val | 0.100 | 0.250 | — | 0 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.667 [0.538, 0.818] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm | on | val | 0.100 | 0.250 | — | 0 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.667 [0.538, 0.818] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm | on | val | 0.100 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | vlm_flash | off | val | 0.100 | 0.250 | — | 0 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.708 [0.692, 0.727] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | val | 0.100 | 0.250 | — | 0 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.708 [0.692, 0.727] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | val | 0.100 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | vlm_flash | on | val | 0.100 | 0.250 | — | 0 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.708 [0.692, 0.727] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | val | 0.100 | 0.250 | — | 0 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.708 [0.692, 0.727] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | val | 0.100 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | val | 0.100 | 0.250 | 0.500 | 108 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.583 [0.538, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | val | 0.100 | 0.250 | 0.500 | 108 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.583 [0.538, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | val | 0.100 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | val | 0.100 | 0.250 | 0.500 | 108 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.583 [0.538, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | val | 0.100 | 0.250 | 0.500 | 108 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.583 [0.538, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | val | 0.100 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | val | 0.100 | 0.250 | 0.500 | 108 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.583 [0.538, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | val | 0.100 | 0.250 | 0.500 | 108 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.583 [0.538, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | val | 0.100 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | val | 0.100 | 0.250 | 0.500 | 108 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.583 [0.538, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | val | 0.100 | 0.250 | 0.500 | 108 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.583 [0.538, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | val | 0.100 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | val | 0.100 | 0.250 | 0.500 | 108 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.542 [0.462, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | val | 0.100 | 0.250 | 0.500 | 108 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.542 [0.462, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | val | 0.100 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | val | 0.100 | 0.250 | 0.500 | 108 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.542 [0.462, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | val | 0.100 | 0.250 | 0.500 | 108 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.542 [0.462, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | val | 0.100 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | val | 0.100 | 0.250 | 0.500 | 108 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.542 [0.462, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | val | 0.100 | 0.250 | 0.500 | 108 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.542 [0.462, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | val | 0.100 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | val | 0.100 | 0.250 | 0.500 | 108 | all | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.542 [0.462, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | val | 0.100 | 0.250 | 0.500 | 108 | sharp-full | 3 | 13.000 [5.000, 19.000] | 8.667 [1.000, 14.000] | 4.333 [4.000, 5.000] | 0.000 [0.000, 0.000] | 0.542 [0.462, 0.636] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | val | 0.100 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | off | val | 0.850 | 0.000 | — | 0 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | — | — | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | off | val | 0.850 | 0.000 | — | 0 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | — | — | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | off | val | 0.850 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | on | val | 0.850 | 0.000 | — | 0 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | — | — | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | on | val | 0.850 | 0.000 | — | 0 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | — | — | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | on | val | 0.850 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | val | 0.850 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.487 [0.400, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.487 [0.400, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | val | 0.850 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.538 [0.450, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.538 [0.450, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | val | 0.850 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | val | 0.850 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.615 [0.550, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.615 [0.550, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.385 [0.000, 0.444] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.385 [0.000, 0.444] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.385 [0.000, 0.444] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.385 [0.000, 0.444] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.385 [0.000, 0.444] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.385 [0.000, 0.444] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.385 [0.000, 0.444] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.385 [0.000, 0.444] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.410 [0.300, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.410 [0.300, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.410 [0.300, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.410 [0.300, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.410 [0.300, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.410 [0.300, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.410 [0.300, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 6.000] | 1.667 [1.000, 3.000] | 2.667 [2.000, 3.000] | 0.000 [0.000, 0.000] | 0.410 [0.300, 1.000] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | off | val | 0.900 | 0.000 | — | 0 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | — | — | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | off | val | 0.900 | 0.000 | — | 0 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | — | — | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | off | val | 0.900 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | on | val | 0.900 | 0.000 | — | 0 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | — | — | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | on | val | 0.900 | 0.000 | — | 0 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | — | — | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | on | val | 0.900 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | val | 0.900 | 0.250 | — | 0 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.515 [0.467, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | val | 0.900 | 0.250 | — | 0 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.515 [0.467, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | val | 0.900 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | val | 0.900 | 0.250 | — | 0 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.545 [0.467, 0.611] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | val | 0.900 | 0.250 | — | 0 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.545 [0.467, 0.611] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | val | 0.900 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | val | 0.900 | 0.250 | — | 0 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.636 [0.611, 0.667] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | val | 0.900 | 0.250 | — | 0 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.636 [0.611, 0.667] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | val | 0.900 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | val | 0.900 | 0.250 | — | 0 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.636 [0.611, 0.667] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | val | 0.900 | 0.250 | — | 0 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.636 [0.611, 0.667] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | val | 0.900 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | val | 0.900 | 0.250 | 0.500 | 108 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | val | 0.900 | 0.250 | 0.500 | 108 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | val | 0.900 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | val | 0.900 | 0.250 | 0.500 | 108 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | val | 0.900 | 0.250 | 0.500 | 108 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | val | 0.900 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | val | 0.900 | 0.250 | 0.500 | 108 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | val | 0.900 | 0.250 | 0.500 | 108 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | val | 0.900 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | val | 0.900 | 0.250 | 0.500 | 108 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | val | 0.900 | 0.250 | 0.500 | 108 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | val | 0.900 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | val | 0.900 | 0.250 | 0.500 | 108 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | val | 0.900 | 0.250 | 0.500 | 108 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | val | 0.900 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | val | 0.900 | 0.250 | 0.500 | 108 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | val | 0.900 | 0.250 | 0.500 | 108 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | val | 0.900 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | val | 0.900 | 0.250 | 0.500 | 108 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | val | 0.900 | 0.250 | 0.500 | 108 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | val | 0.900 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | val | 0.900 | 0.250 | 0.500 | 108 | all | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | val | 0.900 | 0.250 | 0.500 | 108 | sharp-full | 3 | 8.000 [5.000, 14.000] | 3.000 [0.000, 8.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.485 [0.400, 0.556] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | val | 0.900 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | none | off | val | 0.700 | 0.000 | — | 0 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | — | — | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | none | off | val | 0.700 | 0.000 | — | 0 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | — | — | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | none | off | val | 0.700 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | none | on | val | 0.700 | 0.000 | — | 0 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | — | — | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | none | on | val | 0.700 | 0.000 | — | 0 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | — | — | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | none | on | val | 0.700 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | val | 0.700 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.514 [0.421, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | val | 0.700 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.514 [0.421, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | val | 0.700 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | val | 0.700 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.595 [0.421, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | val | 0.700 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.595 [0.421, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | val | 0.700 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | val | 0.700 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.622 [0.579, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | val | 0.700 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.622 [0.579, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | val | 0.700 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | val | 0.700 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.622 [0.579, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | val | 0.700 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.622 [0.579, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | val | 0.700 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | val | 0.700 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | val | 0.700 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | val | 0.700 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | val | 0.700 | 0.250 | — | 0 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | val | 0.700 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | val | 0.700 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | val | 0.700 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.405 [0.000, 0.471] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | val | 0.700 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.405 [0.000, 0.471] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | val | 0.700 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | val | 0.700 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.405 [0.000, 0.471] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | val | 0.700 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.405 [0.000, 0.471] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | val | 0.700 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | val | 0.700 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.405 [0.000, 0.471] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | val | 0.700 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.405 [0.000, 0.471] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | val | 0.700 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | val | 0.700 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.405 [0.000, 0.471] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | val | 0.700 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.405 [0.000, 0.471] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | val | 0.700 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | val | 0.700 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | val | 0.700 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | val | 0.700 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | val | 0.700 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | val | 0.700 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | val | 0.700 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | val | 0.700 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | val | 0.700 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | val | 0.700 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | val | 0.700 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | val | 0.700 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [2.000, 7.000] | 2.333 [1.000, 5.000] | 1.333 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.432 [0.316, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | val | 0.700 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | none | off | val | 0.450 | 0.000 | — | 0 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | — | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | none | off | val | 0.450 | 0.000 | — | 0 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | — | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | none | off | val | 0.450 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | none | on | val | 0.450 | 0.000 | — | 0 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | — | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | none | on | val | 0.450 | 0.000 | — | 0 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | — | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | none | on | val | 0.450 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | val | 0.450 | 0.250 | — | 0 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.643 [0.571, 1.000] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | val | 0.450 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.643 [0.571, 1.000] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | val | 0.450 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | val | 0.450 | 0.250 | — | 0 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.643 [0.571, 1.000] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | val | 0.450 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.643 [0.571, 1.000] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | val | 0.450 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | val | 0.450 | 0.250 | — | 0 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.750 [0.714, 1.000] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | val | 0.450 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.750 [0.714, 1.000] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | val | 0.450 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | val | 0.450 | 0.250 | — | 0 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.750 [0.714, 1.000] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | val | 0.450 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.750 [0.714, 1.000] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | val | 0.450 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | val | 0.450 | 0.250 | 0.000 | 36 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.036 [0.000, 0.071] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | val | 0.450 | 0.250 | 0.000 | 36 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.036 [0.000, 0.071] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | val | 0.450 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | val | 0.450 | 0.250 | 0.000 | 36 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.036 [0.000, 0.071] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | val | 0.450 | 0.250 | 0.000 | 36 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.036 [0.000, 0.071] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | val | 0.450 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | val | 0.450 | 0.250 | 0.000 | 36 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.036 [0.000, 0.071] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | val | 0.450 | 0.250 | 0.000 | 36 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.036 [0.000, 0.071] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | val | 0.450 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | val | 0.450 | 0.250 | 0.000 | 36 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.036 [0.000, 0.071] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | val | 0.450 | 0.250 | 0.000 | 36 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.036 [0.000, 0.071] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | val | 0.450 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | val | 0.450 | 0.250 | 0.000 | 36 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.154] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | val | 0.450 | 0.250 | 0.000 | 36 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.154] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | val | 0.450 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | val | 0.450 | 0.250 | 0.000 | 36 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.154] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | val | 0.450 | 0.250 | 0.000 | 36 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.154] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | val | 0.450 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | val | 0.450 | 0.250 | 0.000 | 36 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.154] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | val | 0.450 | 0.250 | 0.000 | 36 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.154] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | val | 0.450 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | val | 0.450 | 0.250 | 0.000 | 36 | all | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.154] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | val | 0.450 | 0.250 | 0.000 | 36 | sharp-full | 3 | 3.667 [3.000, 5.000] | 1.000 [1.000, 1.000] | 2.667 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.143 [0.000, 0.154] | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | val | 0.450 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | none | off | val | 0.650 | 0.000 | — | 0 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | — | — | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | none | off | val | 0.650 | 0.000 | — | 0 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | — | — | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | none | off | val | 0.650 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | none | on | val | 0.650 | 0.000 | — | 0 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | — | — | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | none | on | val | 0.650 | 0.000 | — | 0 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | — | — | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | none | on | val | 0.650 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | val | 0.650 | 0.250 | — | 0 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.655 [0.571, 1.000] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | val | 0.650 | 0.250 | — | 0 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.655 [0.571, 1.000] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | val | 0.650 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | val | 0.650 | 0.250 | — | 0 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.655 [0.571, 1.000] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | val | 0.650 | 0.250 | — | 0 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.655 [0.571, 1.000] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | val | 0.650 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | val | 0.650 | 0.250 | — | 0 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.759 [0.714, 1.000] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | val | 0.650 | 0.250 | — | 0 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.759 [0.714, 1.000] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | val | 0.650 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | val | 0.650 | 0.250 | — | 0 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.759 [0.714, 1.000] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | val | 0.650 | 0.250 | — | 0 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.759 [0.714, 1.000] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | val | 0.650 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | val | 0.650 | 0.250 | 0.000 | 36 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.034 [0.000, 0.071] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | val | 0.650 | 0.250 | 0.000 | 36 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.034 [0.000, 0.071] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | val | 0.650 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | val | 0.650 | 0.250 | 0.000 | 36 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.034 [0.000, 0.071] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | val | 0.650 | 0.250 | 0.000 | 36 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.034 [0.000, 0.071] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | val | 0.650 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | val | 0.650 | 0.250 | 0.000 | 36 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.034 [0.000, 0.071] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | val | 0.650 | 0.250 | 0.000 | 36 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.034 [0.000, 0.071] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | val | 0.650 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | val | 0.650 | 0.250 | 0.000 | 36 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.034 [0.000, 0.071] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | val | 0.650 | 0.250 | 0.000 | 36 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.034 [0.000, 0.071] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | val | 0.650 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | val | 0.650 | 0.250 | 0.000 | 36 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.172 [0.000, 0.214] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | val | 0.650 | 0.250 | 0.000 | 36 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.172 [0.000, 0.214] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | val | 0.650 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | val | 0.650 | 0.250 | 0.000 | 36 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.172 [0.000, 0.214] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | val | 0.650 | 0.250 | 0.000 | 36 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.172 [0.000, 0.214] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | val | 0.650 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | val | 0.650 | 0.250 | 0.000 | 36 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.172 [0.000, 0.214] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | val | 0.650 | 0.250 | 0.000 | 36 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.172 [0.000, 0.214] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | val | 0.650 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | val | 0.650 | 0.250 | 0.000 | 36 | all | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.172 [0.000, 0.214] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | val | 0.650 | 0.250 | 0.000 | 36 | sharp-full | 3 | 1.000 [0.000, 2.000] | 0.667 [0.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.172 [0.000, 0.214] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | val | 0.650 | 0.250 | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | none | off | val | 0.750 | 0.000 | — | 0 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | — | — | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | none | off | val | 0.750 | 0.000 | — | 0 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | — | — | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | none | off | val | 0.750 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | none | on | val | 0.750 | 0.000 | — | 0 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | — | — | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | none | on | val | 0.750 | 0.000 | — | 0 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | — | — | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | none | on | val | 0.750 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | vlm | off | val | 0.750 | 0.250 | — | 0 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.487 [0.421, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm | off | val | 0.750 | 0.250 | — | 0 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.487 [0.421, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm | off | val | 0.750 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | vlm | on | val | 0.750 | 0.250 | — | 0 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.538 [0.474, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm | on | val | 0.750 | 0.250 | — | 0 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.538 [0.474, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm | on | val | 0.750 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | val | 0.750 | 0.250 | — | 0 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.590 [0.579, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | val | 0.750 | 0.250 | — | 0 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.590 [0.579, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | val | 0.750 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | val | 0.750 | 0.250 | — | 0 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.615 [0.579, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | val | 0.750 | 0.250 | — | 0 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.615 [0.579, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | val | 0.750 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | val | 0.750 | 0.250 | 0.500 | 108 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | val | 0.750 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | val | 0.750 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | val | 0.750 | 0.250 | 0.500 | 108 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | val | 0.750 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | val | 0.750 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | val | 0.750 | 0.250 | 0.500 | 108 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | val | 0.750 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | val | 0.750 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | val | 0.750 | 0.250 | 0.500 | 108 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | val | 0.750 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | val | 0.750 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | val | 0.750 | 0.250 | 0.500 | 108 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | val | 0.750 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | val | 0.750 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | val | 0.750 | 0.250 | 0.500 | 108 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | val | 0.750 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | val | 0.750 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | val | 0.750 | 0.250 | 0.500 | 108 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | val | 0.750 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | val | 0.750 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | val | 0.750 | 0.250 | 0.500 | 108 | all | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | val | 0.750 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.333 [0.000, 8.000] | 3.667 [0.000, 9.000] | 1.667 [1.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.316, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | val | 0.750 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | none | off | val | 0.200 | 0.000 | — | 0 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | — | — | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | none | off | val | 0.200 | 0.000 | — | 0 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | — | — | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | none | off | val | 0.200 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | none | on | val | 0.200 | 0.000 | — | 0 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | — | — | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | none | on | val | 0.200 | 0.000 | — | 0 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | — | — | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | none | on | val | 0.200 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | vlm | off | val | 0.200 | 0.250 | — | 0 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.487 [0.400, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm | off | val | 0.200 | 0.250 | — | 0 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.487 [0.400, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm | off | val | 0.200 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | vlm | on | val | 0.200 | 0.250 | — | 0 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.538 [0.450, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm | on | val | 0.200 | 0.250 | — | 0 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.538 [0.450, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm | on | val | 0.200 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | vlm_flash | off | val | 0.200 | 0.250 | — | 0 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm_flash | off | val | 0.200 | 0.250 | — | 0 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm_flash | off | val | 0.200 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | vlm_flash | on | val | 0.200 | 0.250 | — | 0 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm_flash | on | val | 0.200 | 0.250 | — | 0 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm_flash | on | val | 0.200 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | val | 0.200 | 0.250 | 0.500 | 108 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | val | 0.200 | 0.250 | 0.500 | 108 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | val | 0.200 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | val | 0.200 | 0.250 | 0.500 | 108 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | val | 0.200 | 0.250 | 0.500 | 108 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | val | 0.200 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | val | 0.200 | 0.250 | 0.500 | 108 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | val | 0.200 | 0.250 | 0.500 | 108 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | val | 0.200 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | val | 0.200 | 0.250 | 0.500 | 108 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | val | 0.200 | 0.250 | 0.500 | 108 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | val | 0.200 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | val | 0.200 | 0.250 | 0.500 | 108 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | val | 0.200 | 0.250 | 0.500 | 108 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | val | 0.200 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | val | 0.200 | 0.250 | 0.500 | 108 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | val | 0.200 | 0.250 | 0.500 | 108 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | val | 0.200 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | val | 0.200 | 0.250 | 0.500 | 108 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | val | 0.200 | 0.250 | 0.500 | 108 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | val | 0.200 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | val | 0.200 | 0.250 | 0.500 | 108 | all | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | val | 0.200 | 0.250 | 0.500 | 108 | sharp-full | 3 | 2.000 [1.000, 3.000] | 1.000 [1.000, 1.000] | 1.000 [0.000, 2.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | val | 0.200 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | none | off | val | 0.850 | 0.000 | — | 0 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | — | — | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | none | off | val | 0.850 | 0.000 | — | 0 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | — | — | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | none | off | val | 0.850 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | none | on | val | 0.850 | 0.000 | — | 0 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | — | — | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | none | on | val | 0.850 | 0.000 | — | 0 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | — | — | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | none | on | val | 0.850 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | vlm | off | val | 0.850 | 0.250 | — | 0 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.514 [0.421, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm | off | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.514 [0.421, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm | off | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | vlm | on | val | 0.850 | 0.250 | — | 0 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.568 [0.474, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm | on | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.568 [0.474, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm | on | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | val | 0.850 | 0.250 | — | 0 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.622 [0.579, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.622 [0.579, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | val | 0.850 | 0.250 | — | 0 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.622 [0.579, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.622 [0.579, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.667 [0.000, 6.000] | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.000 [0.000, 0.000] | 0.459 [0.316, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | none | off | val | 0.850 | 0.000 | — | 0 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | — | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | none | off | val | 0.850 | 0.000 | — | 0 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | — | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | none | off | val | 0.850 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | none | on | val | 0.850 | 0.000 | — | 0 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | — | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | none | on | val | 0.850 | 0.000 | — | 0 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | — | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | none | on | val | 0.850 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | val | 0.850 | 0.250 | — | 0 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.528 [0.444, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.528 [0.444, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | val | 0.850 | 0.250 | — | 0 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.583 [0.500, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.583 [0.500, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | val | 0.850 | 0.250 | — | 0 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.639 [0.611, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.639 [0.611, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | val | 0.850 | 0.250 | — | 0 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.639 [0.611, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | val | 0.850 | 0.250 | — | 0 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.639 [0.611, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | val | 0.850 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | all | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | sharp-full | 3 | 4.333 [0.000, 8.000] | 1.667 [1.000, 2.000] | 4.000 [2.000, 6.000] | 0.000 [0.000, 0.000] | 0.472 [0.333, 1.000] | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | val | 0.850 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | none | off | val | 0.050 | 0.000 | — | 0 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | — | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | none | off | val | 0.050 | 0.000 | — | 0 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | — | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | none | off | val | 0.050 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | none | on | val | 0.050 | 0.000 | — | 0 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | — | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | none | on | val | 0.050 | 0.000 | — | 0 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | — | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | none | on | val | 0.050 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | vlm | off | val | 0.050 | 0.250 | — | 0 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.667 [0.625, 0.750] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm | off | val | 0.050 | 0.250 | — | 0 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.667 [0.625, 0.750] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm | off | val | 0.050 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | vlm | on | val | 0.050 | 0.250 | — | 0 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.667 [0.625, 0.750] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm | on | val | 0.050 | 0.250 | — | 0 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.667 [0.625, 0.750] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm | on | val | 0.050 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | val | 0.050 | 0.250 | — | 0 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.750 [0.750, 0.750] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | val | 0.050 | 0.250 | — | 0 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.750 [0.750, 0.750] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | val | 0.050 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | val | 0.050 | 0.250 | — | 0 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.833 [0.750, 0.875] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | val | 0.050 | 0.250 | — | 0 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.833 [0.750, 0.875] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | val | 0.050 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | all | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | sharp-full | 3 | 14.000 [11.000, 18.000] | 9.000 [6.000, 12.000] | 5.000 [4.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | val | 0.050 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | none | off | val | 0.650 | 0.000 | — | 0 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | — | — | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | none | off | val | 0.650 | 0.000 | — | 0 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | — | — | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | none | off | val | 0.650 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | none | on | val | 0.650 | 0.000 | — | 0 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | — | — | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | none | on | val | 0.650 | 0.000 | — | 0 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | — | — | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | none | on | val | 0.650 | 0.000 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | val | 0.650 | 0.250 | — | 0 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.487 [0.400, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | val | 0.650 | 0.250 | — | 0 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.487 [0.400, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | val | 0.650 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | val | 0.650 | 0.250 | — | 0 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.590 [0.450, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | val | 0.650 | 0.250 | — | 0 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.590 [0.450, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | val | 0.650 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | val | 0.650 | 0.250 | — | 0 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | val | 0.650 | 0.250 | — | 0 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | val | 0.650 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | val | 0.650 | 0.250 | — | 0 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | val | 0.650 | 0.250 | — | 0 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.590 [0.550, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | val | 0.650 | 0.250 | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | val | 0.650 | 0.250 | 0.500 | 108 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | val | 0.650 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | val | 0.650 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | val | 0.650 | 0.250 | 0.500 | 108 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | val | 0.650 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | val | 0.650 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | val | 0.650 | 0.250 | 0.500 | 108 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | val | 0.650 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | val | 0.650 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | val | 0.650 | 0.250 | 0.500 | 108 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | val | 0.650 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | val | 0.650 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | val | 0.650 | 0.250 | 0.500 | 108 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | val | 0.650 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | val | 0.650 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | val | 0.650 | 0.250 | 0.500 | 108 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | val | 0.650 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | val | 0.650 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | val | 0.650 | 0.250 | 0.500 | 108 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | val | 0.650 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | val | 0.650 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | val | 0.650 | 0.250 | 0.500 | 108 | all | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | val | 0.650 | 0.250 | 0.500 | 108 | sharp-full | 3 | 3.000 [3.000, 3.000] | 1.000 [0.000, 2.000] | 2.000 [1.000, 3.000] | 0.000 [0.000, 0.000] | 0.436 [0.300, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | val | 0.650 | 0.250 | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |

Per-class count MAE (95% group-bootstrap CI):

| Localiser | Identifier | Pair | Subset | Class | MAE (95% CI) |
|---|---|---|---|---|---|
| B1_owlv2_dinov2-4c99011d | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | nns-sweetener | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | unknown | 19.000 [19.000, 19.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | unknown | 19.000 [19.000, 19.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | nns-sweetener | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | unknown | 19.000 [19.000, 19.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | unknown | 19.000 [19.000, 19.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | unknown | 16.000 [16.000, 16.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | flavor-vanilla | 4.000 [4.000, 4.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | unknown | 14.000 [14.000, 14.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | unknown | 14.000 [14.000, 14.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | flavor-vanilla | 4.000 [4.000, 4.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | unknown | 14.000 [14.000, 14.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | unknown | 14.000 [14.000, 14.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | flavor-vanilla | 4.000 [4.000, 4.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | unknown | 14.000 [14.000, 14.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | unknown | 14.000 [14.000, 14.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | flavor-vanilla | 4.000 [4.000, 4.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | unknown | 14.000 [14.000, 14.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | unknown | 14.000 [14.000, 14.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | unknown | 7.000 [7.000, 7.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | unknown | 7.000 [7.000, 7.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | unknown | 7.000 [7.000, 7.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | unknown | 7.000 [7.000, 7.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | unknown | 6.000 [6.000, 6.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | unknown | 6.000 [6.000, 6.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | unknown | 6.000 [6.000, 6.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | unknown | 6.000 [6.000, 6.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | unknown | 5.000 [5.000, 5.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | unknown | 5.000 [5.000, 5.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | unknown | 5.000 [5.000, 5.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | unknown | 5.000 [5.000, 5.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | unknown | 5.000 [5.000, 5.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | unknown | 5.000 [5.000, 5.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | unknown | 5.000 [5.000, 5.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | unknown | 5.000 [5.000, 5.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | flavor-vanilla | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | unknown | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | flavor-vanilla | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | unknown | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | flavor-vanilla | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | unknown | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | flavor-vanilla | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | unknown | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | flavor-vanilla | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | unknown | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | unknown | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | flavor-vanilla | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | unknown | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 3.000 [3.000, 3.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | unknown | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | unknown | 8.000 [8.000, 8.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | unknown | 8.000 [8.000, 8.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | coca-cola-part1 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | coca-cola-part2 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | coca-cola-part1 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | coca-cola-part2 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | hi-c | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | hi-c | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | hi-c | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | hi-c | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | hi-c | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | hi-c | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | unknown | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | flavor-vanilla | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | nns-sweetener | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | unknown | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | flavor-vanilla | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | flavor-vanilla | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | nns-sweetener | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | unknown | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | flavor-vanilla | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | flavor-vanilla | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | nns-sweetener | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | unknown | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | flavor-vanilla | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | nns-sweetener | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | unknown | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | unknown | 3.000 [3.000, 3.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | unknown | 3.000 [3.000, 3.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | coca-cola-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | coca-cola-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | unknown | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | coca-cola-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | coca-cola-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | hi-c | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | sprite | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | hi-c | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | sprite | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | hi-c | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | sprite | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | hi-c | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | sprite | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | hi-c | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | unknown | 13.000 [13.000, 13.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | unknown | 13.000 [13.000, 13.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | coca-cola-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | coca-cola-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | hi-c | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | unknown | 11.000 [11.000, 11.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | coca-cola-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | coca-cola-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | unknown | 11.000 [11.000, 11.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | nns-sweetener | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | unknown | 13.000 [13.000, 13.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | unknown | 13.000 [13.000, 13.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | nns-sweetener | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | unknown | 13.000 [13.000, 13.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | unknown | 13.000 [13.000, 13.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | hi-c | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | hi-c | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | hi-c | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | hi-c | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | unknown | 9.000 [9.000, 9.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | flavor-vanilla | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | unknown | 7.000 [7.000, 7.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | flavor-vanilla | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | unknown | 7.000 [7.000, 7.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | flavor-vanilla | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | unknown | 7.000 [7.000, 7.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | flavor-vanilla | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | unknown | 7.000 [7.000, 7.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | flavor-vanilla | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | unknown | 7.000 [7.000, 7.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | unknown | 7.000 [7.000, 7.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | flavor-vanilla | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | unknown | 7.000 [7.000, 7.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | unknown | 7.000 [7.000, 7.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | unknown | 13.000 [13.000, 13.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | unknown | 13.000 [13.000, 13.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | unknown | 13.000 [13.000, 13.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | unknown | 13.000 [13.000, 13.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | hi-c | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | unknown | 12.000 [12.000, 12.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | unknown | 12.000 [12.000, 12.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | hi-c | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | unknown | 12.000 [12.000, 12.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | unknown | 12.000 [12.000, 12.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | fanta-orange | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | flavor-vanilla | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | nns-sweetener | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | unknown | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | fanta-orange | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | flavor-vanilla | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | unknown | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | fanta-orange | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | flavor-vanilla | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | nns-sweetener | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | unknown | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | fanta-orange | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | flavor-vanilla | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | unknown | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | fanta-orange | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | flavor-vanilla | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | nns-sweetener | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | unknown | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | fanta-orange | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | unknown | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | fanta-orange | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | flavor-vanilla | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | nns-sweetener | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | unknown | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | fanta-orange | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | unknown | 3.000 [3.000, 3.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | nns-sweetener | 8.000 [8.000, 8.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | unknown | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | nns-sweetener | 8.000 [8.000, 8.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | unknown | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | nns-sweetener | 8.000 [8.000, 8.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | unknown | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | nns-sweetener | 8.000 [8.000, 8.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | unknown | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | nns-sweetener | 8.000 [8.000, 8.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | unknown | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | nns-sweetener | 8.000 [8.000, 8.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | unknown | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | nns-sweetener | 8.000 [8.000, 8.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | unknown | 10.000 [10.000, 10.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | nns-sweetener | 8.000 [8.000, 8.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | unknown | 10.000 [10.000, 10.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | unknown | 8.000 [8.000, 8.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | unknown | 8.000 [8.000, 8.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | unknown | 8.000 [8.000, 8.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | unknown | 8.000 [8.000, 8.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | fanta-orange | 9.000 [9.000, 9.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | fanta-orange | 9.000 [9.000, 9.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | fanta-orange | 9.000 [9.000, 9.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | fanta-orange | 9.000 [9.000, 9.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | fanta-orange | 9.000 [9.000, 9.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | fanta-orange | 9.000 [9.000, 9.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | fanta-orange | 9.000 [9.000, 9.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | fanta-orange | 9.000 [9.000, 9.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | nns-sweetener | 6.000 [6.000, 6.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | nns-sweetener | 6.000 [6.000, 6.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | nns-sweetener | 6.000 [6.000, 6.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | nns-sweetener | 6.000 [6.000, 6.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | nns-sweetener | 6.000 [6.000, 6.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 6.000 [6.000, 6.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | nns-sweetener | 6.000 [6.000, 6.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | unknown | 7.000 [7.000, 7.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 6.000 [6.000, 6.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | unknown | 7.000 [7.000, 7.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | unknown | 15.000 [15.000, 15.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | unknown | 15.000 [15.000, 15.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | coca-cola-part1 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | coca-cola-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | unknown | 11.000 [11.000, 11.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | coca-cola-part1 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | coca-cola-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | unknown | 11.000 [11.000, 11.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | unknown | 12.000 [12.000, 12.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | unknown | 12.000 [12.000, 12.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | unknown | 12.000 [12.000, 12.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | unknown | 12.000 [12.000, 12.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | hi-c | 4.000 [4.000, 4.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | unknown | 9.000 [9.000, 9.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | unknown | 9.000 [9.000, 9.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | hi-c | 4.000 [4.000, 4.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | unknown | 9.000 [9.000, 9.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | unknown | 9.000 [9.000, 9.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | hi-c | 4.000 [4.000, 4.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | unknown | 9.000 [9.000, 9.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | unknown | 9.000 [9.000, 9.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | hi-c | 4.000 [4.000, 4.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | unknown | 9.000 [9.000, 9.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | unknown | 9.000 [9.000, 9.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | hi-c | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | unknown | 13.000 [13.000, 13.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | unknown | 13.000 [13.000, 13.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | hi-c | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | unknown | 13.000 [13.000, 13.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | unknown | 13.000 [13.000, 13.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | hi-c | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | unknown | 13.000 [13.000, 13.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | unknown | 13.000 [13.000, 13.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | hi-c | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | unknown | 13.000 [13.000, 13.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | unknown | 13.000 [13.000, 13.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | unknown | 10.000 [10.000, 10.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | unknown | 10.000 [10.000, 10.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | coca-cola-part1 | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | coca-cola-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | unknown | 4.000 [4.000, 4.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | coca-cola-part1 | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | coca-cola-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | unknown | 4.000 [4.000, 4.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | hi-c | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | unknown | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | hi-c | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | unknown | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | hi-c | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | hi-c | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | hi-c | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | hi-c | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | hi-c | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | hi-c | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | hi-c | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | hi-c | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | hi-c | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | unknown | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | hi-c | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | hi-c | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | unknown | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | hi-c | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | hi-c | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | unknown | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | hi-c | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | hi-c | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | unknown | 6.000 [6.000, 6.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | hi-c | 5.000 [5.000, 5.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | unknown | 7.000 [7.000, 7.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | unknown | 7.000 [7.000, 7.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | coca-cola-part1 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | coca-cola-part2 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | unknown | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | coca-cola-part1 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | coca-cola-part2 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | unknown | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | unknown | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | unknown | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | unknown | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | unknown | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | hi-c | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | unknown | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | hi-c | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | unknown | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | hi-c | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | unknown | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | hi-c | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | unknown | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | hi-c | 4.000 [4.000, 4.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | hi-c | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | unknown | 5.000 [5.000, 5.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | hi-c | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | unknown | 5.000 [5.000, 5.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | hi-c | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | unknown | 5.000 [5.000, 5.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | hi-c | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | unknown | 5.000 [5.000, 5.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | unknown | 8.000 [8.000, 8.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | unknown | 8.000 [8.000, 8.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | coca-cola-part1 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | coca-cola-part2 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | unknown | 4.000 [4.000, 4.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | coca-cola-part1 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | coca-cola-part2 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | unknown | 4.000 [4.000, 4.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | hi-c | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | unknown | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | hi-c | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | unknown | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | hi-c | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | hi-c | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | hi-c | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | hi-c | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | hi-c | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | unknown | 6.000 [6.000, 6.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | hi-c | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | unknown | 6.000 [6.000, 6.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | hi-c | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | unknown | 6.000 [6.000, 6.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | hi-c | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | unknown | 6.000 [6.000, 6.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | unknown | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | unknown | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | coca-cola-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | coca-cola-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | coca-cola-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | coca-cola-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | unknown | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | hi-c | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | flavor-vanilla | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | hi-c | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | flavor-vanilla | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | unknown | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | flavor-vanilla | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | unknown | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | flavor-vanilla | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | unknown | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | flavor-vanilla | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | unknown | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | hi-c | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | unknown | 9.000 [9.000, 9.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | unknown | 9.000 [9.000, 9.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | coca-cola-part1 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | coca-cola-part2 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | hi-c | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | unknown | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | coca-cola-part1 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | coca-cola-part2 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | hi-c | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | unknown | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | hi-c | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | unknown | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | flavor-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | unknown | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | hi-c | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | hi-c | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | hi-c | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | hi-c | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | fanta-orange | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | hi-c | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | hi-c | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | fanta-orange | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | hi-c | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | fanta-orange | 3.000 [3.000, 3.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | hi-c | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | hi-c | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | unknown | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | hi-c | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | hi-c | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | unknown | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | hi-c | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | fanta-orange | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | hi-c | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | unknown | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | hi-c | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | unknown | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | fanta-orange | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | hi-c | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | unknown | 6.000 [6.000, 6.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | hi-c | 5.000 [5.000, 5.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | unknown | 6.000 [6.000, 6.000] |

Best by total count MAE: **G2_rfdetr_small_faces-7c7b4587 / none / pair off** (1.000).

Per-image total-count errors on the usable test images (identifier none, pair off; |predicted - human|):

| Localiser | Score thr | Thr source | 065.jpg | 002.jpg | 045.jpg |
|---|---:|---|---:|---:|---:|
| B1_owlv2_dinov2-4c99011d | 0.050 | val | 3 | 8 | 1 |
| B3_gdino_dinov2-be668339 | 0.100 | val | 19 | 5 | 15 |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | 0.850 | val | 6 | 2 | 3 |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | 0.900 | val | 14 | 5 | 5 |
| D1_rfdetr_small_multiclass-6f3c344b | 0.700 | val | 2 | 7 | 2 |
| G1_rfdetr_medium_faces-cc431fcd | 0.450 | val | 3 | 5 | 3 |
| G2_rfdetr_small_faces-7c7b4587 | 0.650 | val | 2 | 0 | 1 |
| S1_c2_tile768-d52f6d7c | 0.750 | val | 2 | 8 | 0 |
| S2_c2_tile1280-db269936 | 0.200 | val | 2 | 3 | 1 |
| S3_c2_copypaste-205b1c7c | 0.850 | val | 6 | 0 | 5 |
| S5_rfdetr_large_agnostic-7f6457d4 | 0.850 | val | 8 | 0 | 5 |
| S6_owlv2_containment-cbc70383 | 0.050 | val | 13 | 11 | 18 |
| S7_c2_tile1280_copypaste-b6fc182e | 0.650 | val | 3 | 3 | 3 |

Thresholds, exclusions and image lists are in results_gt.json.
