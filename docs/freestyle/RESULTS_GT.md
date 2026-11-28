# Human ground-truth results

Counting is scored against quiz counts; per-class counts come from answered crops.
AP50 compares with human-filtered boxes that originate from model proposals; it is not independent localization GT.
Identity accuracy is conditional on a matched human-answered, known-identity crop.
Class inventory and AP exclude incomplete images and count mismatches.
Thresholds minimize validation total count error, then class count error; embedding validation uses held-out groups.
The validation split needs human count answers only; unanswered validation crops just shrink the identity gallery.
Training-gallery confidence thresholds are selected on validation; +val galleries hold out the query group during validation and use all validation groups at test time. Train and validation may contain near duplicates.
Sharp-full means cabinet_full with no or mild motion blur; partial/blurry means cabinet_partial or severe blur. Other audit cases are omitted from these subsets.

Excluded images:
- val: none
- test: island-cabinet-01/3270fb41-c1f0-4a7d-8497-d35d4745123c/065.jpg (missing_count), island-cabinet-01/50828e4b-1e18-43a5-9c51-a53226bcb264/008.jpg (missing_count)

| Localiser | Identifier | Pair | Train confidence | Train crops | Subset | Images | Total MAE (95% CI) | Top MAE | Bottom MAE | Exact inventory | ID accuracy | AP50 |
|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| B1_owlv2_dinov2-4c99011d | none | off | — | 0 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | — | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | none | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | none | on | — | 0 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | — | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | none | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | vlm | off | — | 0 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | vlm | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | vlm | on | — | 0 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | vlm | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | — | 0 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | — | 0 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | 0.500 | 108 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | 0.500 | 108 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | 0.500 | 108 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | 0.500 | 108 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 5.700 [0.356, 12.900] | 3.300 [0.111, 7.729] | 2.400 [0.000, 5.113] | 0.667 [0.364, 1.000] | — | 0.159 [0.000, 0.437] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.507 [0.507, 0.507] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | none | off | — | 0 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | — | — | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | none | off | — | 0 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | — | — | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | none | on | — | 0 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | — | — | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | none | on | — | 0 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | — | — | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | vlm | off | — | 0 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.667 [0.000, 0.857] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | vlm | off | — | 0 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.667 [0.000, 0.857] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | vlm | on | — | 0 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.778 [0.500, 0.857] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | vlm | on | — | 0 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.778 [0.500, 0.857] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | vlm_flash | off | — | 0 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.778 [0.000, 1.000] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | — | 0 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.778 [0.000, 1.000] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | vlm_flash | on | — | 0 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.778 [0.000, 1.000] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | — | 0 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.778 [0.000, 1.000] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | 0.500 | 108 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.667 [0.000, 0.857] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.667 [0.000, 0.857] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | 0.500 | 108 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.667 [0.000, 0.857] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.667 [0.000, 0.857] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.667 [0.000, 0.857] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.667 [0.000, 0.857] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.667 [0.000, 0.857] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.667 [0.000, 0.857] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | 0.500 | 108 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.556 [0.000, 0.714] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.556 [0.000, 0.714] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | 0.500 | 108 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.556 [0.000, 0.714] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.556 [0.000, 0.714] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.556 [0.000, 0.714] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.556 [0.000, 0.714] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 5.700 [2.000, 11.222] | 3.700 [1.444, 7.223] | 2.000 [0.300, 4.300] | 0.222 [0.000, 0.500] | 0.556 [0.000, 0.714] | 0.154 [0.000, 0.362] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 19.500 [14.000, 25.000] | 11.500 [6.000, 17.000] | 8.000 [8.000, 8.000] | 0.000 [0.000, 0.000] | 0.556 [0.000, 0.714] | 0.522 [0.522, 0.522] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | off | — | 0 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | — | — | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | off | — | 0 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | — | — | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | on | — | 0 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | — | — | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | on | — | 0 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | — | — | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | — | 0 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.500 [0.444, 0.571] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | — | 0 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.500 [0.444, 0.571] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | — | 0 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.531 [0.444, 0.643] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | — | 0 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.531 [0.444, 0.643] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | — | 0 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.625 [0.611, 0.643] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | — | 0 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.625 [0.611, 0.643] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | — | 0 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.625 [0.611, 0.643] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | — | 0 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.625 [0.611, 0.643] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | 0.500 | 108 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.406 [0.389, 0.429] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.406 [0.389, 0.429] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | 0.500 | 108 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.406 [0.389, 0.429] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.406 [0.389, 0.429] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.406 [0.389, 0.429] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.406 [0.389, 0.429] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.406 [0.389, 0.429] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.406 [0.389, 0.429] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | 0.500 | 108 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.406 [0.333, 0.500] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.406 [0.333, 0.500] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | 0.500 | 108 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.406 [0.333, 0.500] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.406 [0.333, 0.500] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.406 [0.333, 0.500] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.406 [0.333, 0.500] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 2.300 [0.500, 4.444] | 1.100 [0.000, 2.333] | 1.200 [0.200, 2.602] | 0.556 [0.143, 0.889] | 0.406 [0.333, 0.500] | 0.780 [0.000, 0.976] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 6.000 [3.000, 9.000] | 1.500 [0.000, 3.000] | 4.500 [3.000, 6.000] | 0.000 [0.000, 0.000] | 0.406 [0.333, 0.500] | 0.983 [0.983, 0.983] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | off | — | 0 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | — | — | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | off | — | 0 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | on | — | 0 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | — | — | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | on | — | 0 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | — | 0 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | — | 0 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | — | 0 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | — | 0 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | — | 0 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | — | 0 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | — | 0 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | — | 0 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | 0.500 | 108 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | 0.500 | 108 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | 0.500 | 108 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | 0.500 | 108 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.900 [0.000, 12.100] | 3.000 [0.000, 7.368] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.706 [0.000, 0.860] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 24.500 [21.000, 28.000] | 15.000 [13.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.889 [0.889, 0.889] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | none | off | — | 0 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | — | — | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | none | off | — | 0 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | — | — | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | none | on | — | 0 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | — | — | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | none | on | — | 0 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | — | — | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | — | 0 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.667 [0.667, 0.667] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | — | 0 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | — | 0 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.833 [0.667, 1.000] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | — | 0 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.833 [0.667, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | — | 0 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.833 [0.667, 1.000] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | — | 0 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.833 [0.667, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | — | 0 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.833 [0.667, 1.000] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | — | 0 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.833 [0.667, 1.000] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | — | 0 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.500 [0.333, 0.667] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | — | 0 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.500 [0.333, 0.667] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | — | 0 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.500 [0.333, 0.667] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | — | 0 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.500 [0.333, 0.667] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | 0.500 | 108 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.333 [0.333, 0.333] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.333 [0.333, 0.333] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | 0.500 | 108 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.333 [0.333, 0.333] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.333 [0.333, 0.333] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.333 [0.333, 0.333] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.333 [0.333, 0.333] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.333 [0.333, 0.333] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.333 [0.333, 0.333] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | 0.500 | 108 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.500 [0.333, 0.667] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.500 [0.333, 0.667] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | 0.500 | 108 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.500 [0.333, 0.667] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.500 [0.333, 0.667] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.500 [0.333, 0.667] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.500 [0.333, 0.667] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.300 [0.000, 10.558] | 2.800 [0.000, 6.914] | 1.500 [0.000, 3.700] | 0.889 [0.625, 1.000] | 0.500 [0.333, 0.667] | 0.537 [0.000, 0.873] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 21.500 [19.000, 24.000] | 14.000 [12.000, 16.000] | 7.500 [7.000, 8.000] | 0.000 [0.000, 0.000] | 0.500 [0.333, 0.667] | 0.894 [0.894, 0.894] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | none | off | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | — | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | none | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | none | on | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | — | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | none | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | 0.000 | 36 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | 0.000 | 36 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | 0.000 | 36 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | 0.000 | 36 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | 0.000 | 36 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | 0.000 | 36 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | 0.000 | 36 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | 0.000 | 36 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | 0.000 | 36 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | 0.000 | 36 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | 0.000 | 36 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | 0.000 | 36 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | 0.000 | 36 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | 0.000 | 36 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | 0.000 | 36 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.318 [0.000, 0.543] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | 0.000 | 36 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.561 [0.561, 0.561] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | none | off | — | 0 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | — | — | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | none | off | — | 0 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | — | — | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | none | on | — | 0 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | — | — | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | none | on | — | 0 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | — | — | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | — | 0 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.714 [0.667, 0.750] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | — | 0 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.714 [0.667, 0.750] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | — | 0 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.714 [0.667, 0.750] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | — | 0 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.714 [0.667, 0.750] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | — | 0 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.810 [0.778, 0.833] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | — | 0 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.810 [0.778, 0.833] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | — | 0 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.810 [0.778, 0.833] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | — | 0 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.810 [0.778, 0.833] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | 0.000 | 36 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.048 [0.000, 0.111] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | 0.000 | 36 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.048 [0.000, 0.111] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | 0.000 | 36 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.048 [0.000, 0.111] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | 0.000 | 36 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.048 [0.000, 0.111] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | 0.000 | 36 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.048 [0.000, 0.111] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | 0.000 | 36 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.048 [0.000, 0.111] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | 0.000 | 36 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.048 [0.000, 0.111] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | 0.000 | 36 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.048 [0.000, 0.111] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | 0.000 | 36 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.190 [0.167, 0.222] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | 0.000 | 36 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.190 [0.167, 0.222] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | 0.000 | 36 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.190 [0.167, 0.222] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | 0.000 | 36 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.190 [0.167, 0.222] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | 0.000 | 36 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.190 [0.167, 0.222] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | 0.000 | 36 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.190 [0.167, 0.222] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | 0.000 | 36 | all | 10 | 2.700 [0.400, 5.889] | 1.100 [0.167, 2.557] | 1.600 [0.000, 3.556] | 0.556 [0.143, 0.889] | 0.190 [0.167, 0.222] | 0.379 [0.000, 0.514] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | 0.000 | 36 | sharp-full | 2 | 10.500 [10.000, 11.000] | 4.500 [3.000, 6.000] | 6.000 [5.000, 7.000] | 0.000 [0.000, 0.000] | 0.190 [0.167, 0.222] | 0.527 [0.527, 0.527] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | 0.000 | 36 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | none | off | — | 0 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | — | — | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | none | off | — | 0 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | — | — | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | none | on | — | 0 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | — | — | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | none | on | — | 0 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | — | — | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | vlm | off | — | 0 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | vlm | off | — | 0 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | vlm | on | — | 0 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | vlm | on | — | 0 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | — | 0 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.750 [0.000, 1.000] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | — | 0 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.750 [0.000, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | — | 0 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.750 [0.000, 1.000] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | — | 0 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.750 [0.000, 1.000] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | 0.500 | 108 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | 0.500 | 108 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | 0.500 | 108 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | 0.500 | 108 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.600 [0.000, 11.447] | 2.900 [0.000, 7.223] | 1.700 [0.000, 4.223] | 0.889 [0.625, 1.000] | 0.500 [0.000, 0.667] | 0.799 [0.000, 0.912] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 23.000 [19.000, 27.000] | 14.500 [12.000, 17.000] | 8.500 [7.000, 10.000] | 0.000 [0.000, 0.000] | 0.500 [0.000, 0.667] | 0.925 [0.925, 0.925] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | none | off | — | 0 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | — | — | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | none | off | — | 0 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | — | — | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | none | on | — | 0 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | — | — | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | none | on | — | 0 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | — | — | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | vlm | off | — | 0 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.586 [0.500, 0.667] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | vlm | off | — | 0 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.586 [0.500, 0.667] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | vlm | on | — | 0 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.586 [0.500, 0.667] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | vlm | on | — | 0 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.586 [0.500, 0.667] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | vlm_flash | off | — | 0 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.655 [0.643, 0.667] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | vlm_flash | off | — | 0 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.655 [0.643, 0.667] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | vlm_flash | on | — | 0 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.690 [0.667, 0.714] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | vlm_flash | on | — | 0 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.690 [0.667, 0.714] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | 0.500 | 108 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.517 [0.429, 0.600] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.517 [0.429, 0.600] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | 0.500 | 108 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.517 [0.429, 0.600] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.517 [0.429, 0.600] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.517 [0.429, 0.600] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.517 [0.429, 0.600] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.517 [0.429, 0.600] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.517 [0.429, 0.600] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | 0.500 | 108 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.517 [0.429, 0.600] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.517 [0.429, 0.600] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | 0.500 | 108 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.517 [0.429, 0.600] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.517 [0.429, 0.600] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.517 [0.429, 0.600] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.517 [0.429, 0.600] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 2.100 [0.000, 5.111] | 0.700 [0.000, 1.667] | 1.400 [0.000, 3.457] | 0.778 [0.444, 1.000] | 0.517 [0.429, 0.600] | 0.875 [0.000, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 10.000 [8.000, 12.000] | 3.000 [2.000, 4.000] | 7.000 [6.000, 8.000] | 0.000 [0.000, 0.000] | 0.517 [0.429, 0.600] | 0.950 [0.950, 0.950] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | none | off | — | 0 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | — | — | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | none | off | — | 0 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | — | — | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | none | on | — | 0 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | — | — | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | none | on | — | 0 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | — | — | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | vlm | off | — | 0 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.800 [0.800, 0.800] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | vlm | off | — | 0 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.800 [0.800, 0.800] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | vlm | on | — | 0 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.800 [0.800, 0.800] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | vlm | on | — | 0 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.800 [0.800, 0.800] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | — | 0 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | — | 0 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | — | 0 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | — | 0 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | 0.500 | 108 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.600 [0.600, 0.600] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.600, 0.600] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | 0.500 | 108 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.600 [0.600, 0.600] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.600, 0.600] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.600 [0.600, 0.600] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.600, 0.600] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.600 [0.600, 0.600] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.600, 0.600] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | 0.500 | 108 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.600 [0.600, 0.600] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.600, 0.600] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | 0.500 | 108 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.600 [0.600, 0.600] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.600, 0.600] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.600 [0.600, 0.600] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.600, 0.600] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.500 [0.000, 11.235] | 3.000 [0.000, 7.368] | 1.500 [0.000, 4.111] | 0.889 [0.625, 1.000] | 0.600 [0.600, 0.600] | 0.759 [0.000, 0.918] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 22.500 [17.000, 28.000] | 15.000 [13.000, 17.000] | 7.500 [4.000, 11.000] | 0.000 [0.000, 0.000] | 0.600 [0.600, 0.600] | 0.964 [0.964, 0.964] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | none | off | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | — | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | none | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | none | on | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | — | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | none | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | — | 0 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | 0.500 | 108 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | 0.500 | 108 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | 0.500 | 108 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | 0.500 | 108 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 5.000 [0.000, 12.228] | 3.100 [0.000, 7.600] | 1.900 [0.000, 4.729] | 0.889 [0.625, 1.000] | — | 0.772 [0.000, 0.943] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.970 [0.970, 0.970] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | none | off | — | 0 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | — | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | none | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | none | on | — | 0 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | — | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | none | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | — | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | vlm | off | — | 0 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | vlm | on | — | 0 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | — | 0 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | — | 0 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | — | 0 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | 0.500 | 108 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | 0.500 | 108 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | 0.500 | 108 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | 0.500 | 108 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 5.600 [0.000, 12.802] | 3.200 [0.000, 7.667] | 2.400 [0.000, 5.113] | 0.778 [0.444, 1.000] | — | 0.147 [0.000, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 25.000 [22.000, 28.000] | 15.500 [14.000, 17.000] | 9.500 [8.000, 11.000] | 0.000 [0.000, 0.000] | — | 0.357 [0.357, 0.357] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | none | off | — | 0 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | — | — | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | none | off | — | 0 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | — | — | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | none | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | none | on | — | 0 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | — | — | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | none | on | — | 0 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | — | — | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | none | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | — | 0 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.833 [0.833, 0.833] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | — | 0 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.833 [0.833, 0.833] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | — | 0 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.833 [0.833, 0.833] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | — | 0 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.833 [0.833, 0.833] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | — | 0 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | — | 0 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | — | 0 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 1.000 [1.000, 1.000] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | — | 0 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | — | 0 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | 0.500 | 108 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.667 [0.667, 0.667] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | 0.500 | 108 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.667 [0.667, 0.667] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.667 [0.667, 0.667] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.667 [0.667, 0.667] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | 0.500 | 108 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.667 [0.667, 0.667] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | 0.500 | 108 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | 0.500 | 108 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.667 [0.667, 0.667] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | 0.500 | 108 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | 0.500 | 108 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.667 [0.667, 0.667] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | 0.500 | 108 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | 0.500 | 108 | all | 10 | 4.400 [0.000, 11.209] | 3.000 [0.000, 7.368] | 1.400 [0.000, 4.000] | 0.889 [0.625, 1.000] | 0.667 [0.667, 0.667] | 0.767 [0.000, 0.934] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | 0.500 | 108 | sharp-full | 2 | 22.000 [16.000, 28.000] | 15.000 [13.000, 17.000] | 7.000 [3.000, 11.000] | 0.000 [0.000, 0.000] | 0.667 [0.667, 0.667] | 0.964 [0.964, 0.964] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | 0.500 | 108 | partial/blurry | 0 | — | — | — | — | — | — |

Per-class count MAE (95% group-bootstrap CI):

| Localiser | Identifier | Pair | Subset | Class | MAE (95% CI) |
|---|---|---|---|---|---|
| B1_owlv2_dinov2-4c99011d | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | sprite | 0.222 [0.000, 0.556] |
| B1_owlv2_dinov2-4c99011d | vlm | off | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | sprite | 0.222 [0.000, 0.556] |
| B1_owlv2_dinov2-4c99011d | vlm | on | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | sprite | 0.222 [0.000, 0.556] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | sprite | 0.222 [0.000, 0.556] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | vlm_flash | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | nns-sweetener | 0.667 [0.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | nns-sweetener | 0.667 [0.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | nns-sweetener | 0.667 [0.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | nns-sweetener | 0.667 [0.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | nns-sweetener | 0.667 [0.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | nns-sweetener | 0.667 [0.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | nns-sweetener | 0.667 [0.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | nns-sweetener | 0.667 [0.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | all | unknown | 0.889 [0.000, 2.250] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B1_owlv2_dinov2-4c99011d | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | flavor-orange | 0.222 [0.000, 0.444] |
| B3_gdino_dinov2-be668339 | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | nns-sweetener | 0.556 [0.000, 1.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | off | all | unknown | 1.444 [0.625, 2.273] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | flavor-orange | 0.222 [0.000, 0.444] |
| B3_gdino_dinov2-be668339 | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | nns-sweetener | 0.556 [0.000, 1.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm | on | all | unknown | 1.444 [0.625, 2.273] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | flavor-vanilla | 0.222 [0.000, 0.444] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | nns-sweetener | 0.556 [0.000, 1.250] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | sprite | 0.222 [0.000, 0.571] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | all | unknown | 1.333 [0.625, 2.201] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | flavor-vanilla | 0.222 [0.000, 0.444] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | nns-sweetener | 0.556 [0.000, 1.250] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | sprite | 0.222 [0.000, 0.571] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | all | unknown | 1.333 [0.625, 2.201] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | vlm_flash | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.273] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | nns-sweetener | 0.667 [0.000, 1.667] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | all | unknown | 1.333 [0.444, 2.201] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.273] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | nns-sweetener | 0.667 [0.000, 1.667] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | all | unknown | 1.333 [0.444, 2.201] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.273] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | nns-sweetener | 0.667 [0.000, 1.667] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | all | unknown | 1.333 [0.444, 2.201] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.273] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | nns-sweetener | 0.667 [0.000, 1.667] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | all | unknown | 1.333 [0.444, 2.201] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | knn_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | flavor-vanilla | 0.333 [0.000, 0.667] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | nns-sweetener | 0.667 [0.000, 1.667] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | all | unknown | 1.222 [0.444, 2.125] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | flavor-vanilla | 0.333 [0.000, 0.667] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | nns-sweetener | 0.667 [0.000, 1.667] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | all | unknown | 1.222 [0.444, 2.125] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | flavor-vanilla | 0.333 [0.000, 0.667] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | nns-sweetener | 0.667 [0.000, 1.667] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | all | unknown | 1.222 [0.444, 2.125] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | flavor-vanilla | 0.333 [0.000, 0.667] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | nns-sweetener | 0.667 [0.000, 1.667] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | all | unknown | 1.222 [0.444, 2.125] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 0.000 [0.000, 0.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| B3_gdino_dinov2-be668339 | linear_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | fanta-orange | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | flavor-vanilla | 0.333 [0.000, 0.714] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | hi-c | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | nns-sweetener | 0.667 [0.000, 1.625] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | all | unknown | 0.667 [0.125, 1.444] |
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
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | off | sharp-full | unknown | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | fanta-orange | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | flavor-vanilla | 0.333 [0.000, 0.714] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | hi-c | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | nns-sweetener | 0.667 [0.000, 1.625] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | all | unknown | 0.667 [0.125, 1.444] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
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
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm | on | sharp-full | unknown | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | fanta-orange | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | flavor-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | hi-c | 0.333 [0.000, 0.875] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | nns-sweetener | 0.778 [0.000, 1.802] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | all | unknown | 0.333 [0.000, 0.714] |
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
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | fanta-orange | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | flavor-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | hi-c | 0.333 [0.000, 0.875] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | nns-sweetener | 0.778 [0.000, 1.802] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | all | unknown | 0.333 [0.000, 0.714] |
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
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | hi-c | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | vlm_flash | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | hi-c | 0.333 [0.000, 1.125] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | nns-sweetener | 1.111 [0.000, 2.601] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | sprite | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | all | unknown | 0.444 [0.111, 0.857] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | hi-c | 0.333 [0.000, 1.125] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | nns-sweetener | 1.111 [0.000, 2.601] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | sprite | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | all | unknown | 0.444 [0.111, 0.857] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | hi-c | 0.333 [0.000, 1.125] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | nns-sweetener | 1.111 [0.000, 2.601] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | sprite | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | all | unknown | 0.444 [0.111, 0.857] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | hi-c | 0.333 [0.000, 1.125] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | nns-sweetener | 1.111 [0.000, 2.601] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | sprite | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | all | unknown | 0.444 [0.111, 0.857] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | hi-c | 3.000 [3.000, 3.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | knn_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | flavor-vanilla | 0.444 [0.000, 1.500] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | nns-sweetener | 1.333 [0.000, 2.857] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | all | unknown | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | flavor-vanilla | 0.444 [0.000, 1.500] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | nns-sweetener | 1.333 [0.000, 2.857] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | all | unknown | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | flavor-vanilla | 0.444 [0.000, 1.500] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | nns-sweetener | 1.333 [0.000, 2.857] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | flavor-vanilla | 0.444 [0.000, 1.500] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | nns-sweetener | 1.333 [0.000, 2.857] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.556] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 4.000 [4.000, 4.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C1_rfdetr_small_agnostic_dinov2-55ef8ae7 | linear_trainvlm+val | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | vlm_flash | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| C2_rfdetr_medium_agnostic_siglip2-89fe7c2b | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | all | unknown | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | all | unknown | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | all | unknown | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | all | unknown | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | vlm_flash | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | all | unknown | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | all | unknown | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | own | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | all | unknown | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | all | unknown | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | all | unknown | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | all | unknown | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | knn_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| D1_rfdetr_small_multiclass-6f3c344b | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | vlm_flash | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G1_rfdetr_medium_faces-cc431fcd | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | flavor-vanilla | 0.333 [0.000, 0.902] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | nns-sweetener | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | all | unknown | 0.333 [0.000, 0.875] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | flavor-vanilla | 0.333 [0.000, 0.902] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | nns-sweetener | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | all | unknown | 0.333 [0.000, 0.875] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | flavor-vanilla | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | hi-c | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | nns-sweetener | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | all | unknown | 0.444 [0.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | flavor-vanilla | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | hi-c | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | nns-sweetener | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | all | unknown | 0.444 [0.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | vlm_flash | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | fanta-orange | 0.556 [0.000, 1.875] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | flavor-lemon | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | flavor-vanilla | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | hi-c | 0.222 [0.000, 0.556] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | nns-sweetener | 0.444 [0.000, 1.286] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | all | unknown | 0.333 [0.000, 0.714] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | fanta-orange | 5.000 [5.000, 5.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | fanta-orange | 0.556 [0.000, 1.875] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | flavor-lemon | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | flavor-vanilla | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | hi-c | 0.222 [0.000, 0.556] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | nns-sweetener | 0.444 [0.000, 1.286] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | all | unknown | 0.333 [0.000, 0.714] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | fanta-orange | 5.000 [5.000, 5.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | fanta-orange | 0.556 [0.000, 1.875] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | flavor-lemon | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | flavor-vanilla | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | hi-c | 0.222 [0.000, 0.556] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | nns-sweetener | 0.444 [0.000, 1.286] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | all | unknown | 0.333 [0.000, 0.714] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | fanta-orange | 5.000 [5.000, 5.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | fanta-orange | 0.556 [0.000, 1.875] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | flavor-lemon | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | flavor-vanilla | 0.222 [0.000, 0.571] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | hi-c | 0.222 [0.000, 0.556] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | nns-sweetener | 0.444 [0.000, 1.286] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | all | unknown | 0.333 [0.000, 0.714] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | fanta-orange | 5.000 [5.000, 5.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | knn_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | nns-sweetener | 0.556 [0.000, 1.500] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | all | unknown | 0.333 [0.000, 0.714] |
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
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | nns-sweetener | 0.556 [0.000, 1.500] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | all | unknown | 0.333 [0.000, 0.714] |
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
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | nns-sweetener | 0.556 [0.000, 1.500] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | all | unknown | 0.333 [0.000, 0.714] |
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
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | nns-sweetener | 0.556 [0.000, 1.500] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | all | unknown | 0.333 [0.000, 0.714] |
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
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| G2_rfdetr_small_faces-7c7b4587 | linear_trainvlm+val | on | sharp-full | unknown | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | all | unknown | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | all | unknown | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | hi-c | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | all | unknown | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | hi-c | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | all | unknown | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | vlm_flash | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | hi-c | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | hi-c | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | all | unknown | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | all | unknown | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | all | unknown | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | all | unknown | 0.111 [0.000, 0.375] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S1_c2_tile768-d52f6d7c | linear_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | off | all | nns-sweetener | 0.222 [0.000, 0.556] |
| S2_c2_tile1280-db269936 | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | on | all | nns-sweetener | 0.222 [0.000, 0.556] |
| S2_c2_tile1280-db269936 | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | nns-sweetener | 0.333 [0.000, 0.889] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | sprite | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | off | all | unknown | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | off | sharp-full | unknown | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | nns-sweetener | 0.333 [0.000, 0.889] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | sprite | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | on | all | unknown | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | diet-coke-part1 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | diet-coke-part2 | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | vlm_flash | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | fanta-orange | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | nns-sweetener | 0.333 [0.000, 0.889] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | all | unknown | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | fanta-orange | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | nns-sweetener | 0.333 [0.000, 0.889] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | all | unknown | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | fanta-orange | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | nns-sweetener | 0.333 [0.000, 0.889] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | all | unknown | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | fanta-orange | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | nns-sweetener | 0.333 [0.000, 0.889] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | all | unknown | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | fanta-orange | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | sprite | 0.000 [0.000, 0.000] |
| S2_c2_tile1280-db269936 | knn_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | fanta-orange | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | nns-sweetener | 0.222 [0.000, 0.556] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | all | unknown | 0.111 [0.000, 0.375] |
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
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | fanta-orange | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | nns-sweetener | 0.222 [0.000, 0.556] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | all | unknown | 0.111 [0.000, 0.375] |
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
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | fanta-orange | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | nns-sweetener | 0.222 [0.000, 0.556] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | all | unknown | 0.111 [0.000, 0.375] |
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
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | off | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | fanta-orange | 0.222 [0.000, 0.750] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | nns-sweetener | 0.222 [0.000, 0.556] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | all | unknown | 0.111 [0.000, 0.375] |
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
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S2_c2_tile1280-db269936 | linear_trainvlm+val | on | sharp-full | unknown | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | hi-c | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | hi-c | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | vlm_flash | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | nns-sweetener | 0.333 [0.000, 1.125] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | nns-sweetener | 3.000 [3.000, 3.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S3_c2_copypaste-205b1c7c | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | vlm_flash | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S5_rfdetr_large_agnostic-7f6457d4 | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | sprite | 0.222 [0.000, 0.556] |
| S6_owlv2_containment-cbc70383 | vlm | off | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | sprite | 0.222 [0.000, 0.556] |
| S6_owlv2_containment-cbc70383 | vlm | on | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | sprite | 0.222 [0.000, 0.556] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | nns-sweetener | 0.556 [0.000, 1.875] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | sprite | 0.222 [0.000, 0.556] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | vlm_flash | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | nns-sweetener | 0.667 [0.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | nns-sweetener | 0.667 [0.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | nns-sweetener | 0.667 [0.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | nns-sweetener | 0.667 [0.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | nns-sweetener | 0.667 [0.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | nns-sweetener | 0.667 [0.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | nns-sweetener | 0.667 [0.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | nns-sweetener | 0.667 [0.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | all | unknown | 0.778 [0.000, 2.143] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | nns-sweetener | 5.000 [5.000, 5.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S6_owlv2_containment-cbc70383 | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | flavor-lemon | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | flavor-lemon | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | hi-c | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | flavor-vanilla | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | hi-c | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | fanta-orange | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | flavor-vanilla | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | hi-c | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | ingredient-aa | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | vlm_flash | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | knn_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm | on | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | fanta-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | off | sharp-full | unknown | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | coca-cola-part2 | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | diet-coke-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | diet-coke-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | dr-pepper-part1 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | dr-pepper-part2 | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | fanta-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | flavor-cherry | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | flavor-lemon | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | flavor-orange | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | hi-c | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | ingredient-aa | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | ingredient-cc | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | nns-sweetener | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | sprite | 0.111 [0.000, 0.375] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | all | unknown | 0.222 [0.000, 0.750] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | coca-cola-part2 | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | diet-coke-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | diet-coke-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | dr-pepper-part1 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | dr-pepper-part2 | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | fanta-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | flavor-cherry | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | flavor-lemon | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | flavor-orange | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | flavor-vanilla | 0.000 [0.000, 0.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | hi-c | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | ingredient-aa | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | ingredient-cc | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | nns-sweetener | 2.000 [2.000, 2.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | sprite | 1.000 [1.000, 1.000] |
| S7_c2_tile1280_copypaste-b6fc182e | linear_trainvlm+val | on | sharp-full | unknown | 2.000 [2.000, 2.000] |

Best by total count MAE: **S2_c2_tile1280-db269936 / none / pair off** (2.100).

Thresholds, exclusions and image lists are in results_gt.json.
