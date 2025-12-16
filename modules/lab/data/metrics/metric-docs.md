### Metrics Filenames

FILENAME FORMAT for all types, except val from hyperparam search trial: `model_embedding_lg_type_seed_yyyy-MM-dd-hh-mm-ss.json`
(`_lg` included only if using large dataset; `model_embedding` can be substituted by `finetuned-mbert` even though it is just the model, because finetuned-bert does not use embeddings)

FILENAME FORMAT for val from hyperparam search trial: `model_embedding_lg_val_tX_seed_yyyy-MM-dd-hh-mm-ss.json`
(`_lg` included only if using large dataset; `model_embedding` can be substituted by `finetuned-mbert` even though it is just the model, because finetuned-bert does not use embeddings; `X` is an integer)

FILENAME EXAMPLE 1: `finetuned-mbert_train_42_2025-12-02-20-05-43.json`
FILENAME EXAMPLE 2: `svm_minilm_lg_val_t1_42_2025-12-02-20-05-43.json`

### All Metric File Structures
1. TRAIN
```json
{
  "model": "model_embedding_lg",          // without seed and type in identifier; `model_embedding` can be substituted by `finetuned-mbert` even though it is just the model, because finetuned-bert does not use embeddings
  "params": { hyperparameter: 0 },        // including batch_size for fine-tuned model
  "type": "train",                        // always "train"
  "seed": 42,                             // int
  "time_start": "2025-12-15T10:30:00",    // ISO 8601 timestamp string
  "time_end": "2025-12-15T10:45:00",      // ISO 8601 timestamp string
  "mse": 0.75,                            // float
  "accuracy": 0.5,                        // float
  "precision": [<float>, ...],            // list[float]: Per-label precision (length 6)
  "recall": [<float>, ...],               // list[float]: Per-label recall (length 6)
  "f1": [<float>, ...],                   // list[float]: Per-label F1 score (length 6)
  "support": [<int>, ...],                // list[int]: Per-label sample count (length 6)
  "confusion_matrix": [[<int>, ...], ...], // list[list[int]]: 6x6 confusion matrix
  "train_losses": [<float>, ...],         // list[float] Per-batch, finetuned-mbert model only
  "epoch_batch_counts": [<int>, ...]     // list[int]: Number of batches per epoch, finetuned-mbert model only
}
```

2. TEST
```json
{
  "model": "model_embedding_lg",          // without seed and type in identifier; `model_embedding` can be substituted by `finetuned-mbert` even though it is just the model, because finetuned-bert does not use embeddings
  "params": { hyperparameter: 0 },        // including batch_size for fine-tuned model
  "type": "test",                         // always "test"
  "seed": 42,                             // int
  "time_start": "2025-12-15T10:30:00",    // ISO 8601 timestamp string
  "time_end": "2025-12-15T10:45:00",      // ISO 8601 timestamp string
  "mse": 0.75,                            // float
  "accuracy": 0.5,                        // float
  "precision": [<float>, ...],            // list[float]: Per-label precision (length 6)
  "recall": [<float>, ...],               // list[float]: Per-label recall (length 6)
  "f1": [<float>, ...],                   // list[float]: Per-label F1 score (length 6)
  "support": [<int>, ...],                // list[int]: Per-label sample count (length 6)
  "confusion_matrix": [[<int>, ...], ...] // list[list[int]]: 6x6 confusion matrix
}
```

3. VALIDATION (CV)
```json
{
  "model": "model_embedding_lg",              // without seed and type in identifier; `model_embedding` can be substituted by `finetuned-mbert` even though it is just the model, because finetuned-bert does not use embeddings
  "params": { hyperparameter: 0 },            // including batch_size for fine-tuned model
  "type": "val",                              // always "val"
  "seed": 42,                                 // int
  "time_start": "2025-12-15T10:30:00",        // ISO 8601 timestamp string
  "time_end": "2025-12-15T10:45:00",          // ISO 8601 timestamp string
  "folds": [
    {
      "mse": 0.75,                            // float
      "accuracy": 0.5,                        // float
      "precision": [<float>, ...],            // list[float]: Per-label precision (length 6)
      "recall": [<float>, ...],               // list[float]: Per-label recall (length 6)
      "f1": [<float>, ...],                   // list[float]: Per-label F1 score (length 6)
      "support": [<int>, ...],                // list[int]: Per-label sample count (length 6)
      "confusion_matrix": [[<int>, ...], ...], // list[list[int]]: 6x6 confusion matrix
      "train_losses": [<float>, ...],         // list[float] Per-batch, finetuned-mbert model only
      "epoch_batch_counts": [<int>, ...],    // list[int]: Number of batches per epoch, finetuned-mbert model only
      "val_losses": [<float>, ...]            // list[float] Per-epoch, finetuned-mbert model only
    }
  ],
  "trial": 1,                                 // int, only if its exported from a hyperparam search trial
}
```

### Values that are supposed to be computed on-demand and are NOT stored
- best epoch by val loss, best val loss (`common.py::get_best_epoch`)
- metrics (mse, accuracy, precision, recall, f1, support, confusion_matrix) averaged across all folds (`common.py::average_metrics`)
- average of metrics for multiple seeds (in different files)
