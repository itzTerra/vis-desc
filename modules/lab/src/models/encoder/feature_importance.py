import numpy as np
from text2features import FeatureExtractorPipeline
from training import MODEL_PARAMS, TRAINER_CLASSES
from models.encoder.common import _average_metrics

model = "catboost"
embeddings = "minilm"
params = MODEL_PARAMS[model][embeddings]
include_large = False
save_model = True
seeds = [40 + i for i in range(3)]  # 40,41,42

feature_count = FeatureExtractorPipeline.FEATURE_COUNT
extractor_counts = FeatureExtractorPipeline.EXTRACTOR_FEATURE_COUNTS
trainer_class = TRAINER_CLASSES[model]
MINILM_DIM = 384


def run_config(base_label, feature_mask=None, minilm_mask=None):
    per_seed = []
    for seed in seeds:
        trainer = trainer_class(
            params,
            embeddings=embeddings,
            include_large=include_large,
            enable_train=True,
            enable_test=True,
            save_model=save_model,
            feature_mask=feature_mask,
            minilm_mask=minilm_mask,
            label=f"{base_label}-s{seed}",
            seed=seed,
        )
        metrics = trainer.run_full_training()
        per_seed.append(metrics)
    averaged = _average_metrics(per_seed)
    return averaged, per_seed


def print_metrics(label, avg_metrics, per_seed_metrics):
    print(f"Config: {label}")
    print("Averaged Metrics:")
    print(avg_metrics)


# 1. minilm + features (all)
# trainer = trainer_class(
#     params,
#     embeddings=embeddings,
#     include_large=include_large,
#     enable_train=True,
#     enable_test=True,
#     save_model=save_model,
#     feature_mask=None,
#     minilm_mask=None,
# )
# metrics = trainer.run_full_training()
# results.append(("all", metrics))

# 2. features only (no minilm)
label = f"features-{embeddings}"
avg_metrics, per_seed_metrics = run_config(
    label, feature_mask=None, minilm_mask=np.zeros(MINILM_DIM, dtype=bool)
)
print_metrics(label, avg_metrics, per_seed_metrics)

# 3. minilm+features minus each extractor group (as defined in EXTRACTOR_FEATURE_COUNTS)
feature_ranges: dict[str, tuple[int, int]] = {}
offset = 0
for name, count in extractor_counts.items():
    feature_ranges[name] = (offset, offset + count)
    offset += count

assert offset == feature_count, (
    f"Offset {offset} does not match FEATURE_COUNT {feature_count}"
)

for name, (start, end) in feature_ranges.items():
    label = f"features-{name}"
    mask = np.ones(feature_count, dtype=bool)
    mask[start:end] = False
    avg_metrics, per_seed_metrics = run_config(
        label, feature_mask=mask, minilm_mask=None
    )
    print_metrics(label, avg_metrics, per_seed_metrics)
