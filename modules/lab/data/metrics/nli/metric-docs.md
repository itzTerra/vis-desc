### NLI Zero-Shot Model Metrics Filenames

FILENAME FORMAT: `nli_dataset_yyyy-MM-dd-hh-mm-ss.json`
- `dataset`: train or test
- Timestamp: ISO 8601 format (yyyy-MM-dd-hh-mm-ss)

FILENAME EXAMPLES:
- `nli_train_2025-12-21-14-30-45.json`
- `nli_test_2025-12-21-15-45-20.json`

### Metric File Structure

```json
{
  "dataset": "train",                          // string: "train" or "test"
  "hypothesis_template": "This text is {} in terms of visual details of characters, setting, or environment.",  // string: Template used for zero-shot classification
  "candidate_labels": ["not detailed", "detailed"],  // list[string]: Labels used in classification, MUST BE ORDERED UNIFORMLY IN THE 0-1 REGRESSION SENSE
  "models": [                                  // list[object]: Evaluation results for each model
    {
      "model_name": "MoritzLaurer/roberta-base-zeroshot-v2.0-c",  // string: Full model identifier (HuggingFace model ID or custom name)
      "probabilities": [                       // list[list[float]]: Probabilities for each sample (outer list) and each label (inner list)
        [0.8234, 0.1766],                      // Sample 1: [prob_label_0, prob_label_1]
        [0.6521, 0.3479],                      // Sample 2: [prob_label_0, prob_label_1]
        [0.9102, 0.0898]                       // Sample 3: [prob_label_0, prob_label_1]
      ],
      "scores": [0.1766, 0.3479, 0.0898],      // list[float]: Sample scores computed from label probabilities
      "corr": 0.3512,                  // float: Pearson correlation coefficient of the sample scores and train set labels
      "device": "NVIDIA GeForce RTX 3090",     // string: Device name where evaluation was performed
      "batch_size": 16,                        // int: Number of samples per batch
      "performance": {                         // object: Performance metrics
        "throughput": 125.5,                   // float: Samples per second
        "latency_mean": 7.97,                  // float: Average time to process one sample (milliseconds)
        "latency_std": 0.42,                   // float: Standard deviation of latency in milliseconds
      },
      "time_start": "2025-12-21T14:30:45",        // string: ISO 8601 timestamp when evaluation started
      "time_end": "2025-12-21T14:35:12",          // string: ISO 8601 timestamp when evaluation completed
    }
  ]
}
```
