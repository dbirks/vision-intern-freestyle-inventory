# Dense scenes

For large frames with many small objects, start with tiles and padded judge crops:

```python
from workflows.vlm_label import label_dataset
from workflows.vlm_judge import judge_labels

label_dataset("images", ["object"], "labels.json", backend="openai",
              tile=1024, tile_overlap=0.2, max_dimension=None,
              image_format="png", max_tokens=8192, n_samples=2, recursive=True)
judge_labels("images", "judged.json", annotations="labels.json", backend="openai",
             judge_mode="crops", judge_batch_size=12, max_dimension=None,
             image_format="png", max_tokens=4096)
```

Configure separate worker model IDs and endpoints for each stage. Check truncation
flags and unjudged verdicts before using the output; agreement scores are sample
fractions, not calibrated probabilities. Tiling and repeated samples increase calls.
