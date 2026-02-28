## Lab

This is the environment used for dataset creation, feature engineering, model training.

### File Structure
Core files in sensible running order:

```txt
.
├── data/
└── src/
    ├── dataset_small/
    │   ├── book_collection.ipynb
    │   ├── book_preprocessing.py
    │   ├── book_segmenting.py
    │   ├── segment_sampling.ipynb
    │   ├── agreement.py
    │   └── dataset_small.ipynb
    ├── dataset_referential_and_lg_heuristic.ipynb
    ├── dataset_large.ipynb
    ├── dataset_concreteness.ipynb
    ├── models/
    │   ├── encoder/
    │   │   ├── feature_extractors.ipynb
    │   │   ├── text2features.py
    │   │   ├── data_preparation.ipynb
    │   │   ├── hyperparam_search.py
    │   │   └── train.py
    │   ├── nli/
    │   │   └── run.py
    │   └── llm/
    │       ├── optimize_prompts.py
    │       └── run.py
    └── evaluation/
        ├── encoder/
        │   ├── encoder.ipynb
        │   └── feature_importance.ipynb
        ├── nli/
        │   └── nli.ipynb
        └── llm/
            └── llm.ipynb
```
