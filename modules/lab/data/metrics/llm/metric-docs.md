### LLM Model Metrics Filenames

FILENAME FORMAT: `promptId_yyyy-MM-dd-hh-mm-ss.json`
- `promptId`: result of Prompt.get_id()
- Timestamp: ISO 8601 format (yyyy-MM-dd-hh-mm-ss)

FILENAME EXAMPLE:
- `zero-shot_action_cot_Rate-this-text_2025-12-21-14-30-45.json`

### Metric File Structure

```json
{
  "system": "System prompt",                    // string
  "prompt": "Full prompt text",                 // string
  "prompt_token_count": 40,                     // int
  "models": [                                  // list[object]: Evaluation results for each model
    {
      "model_name": "MoritzLaurer/roberta-base-zeroshot-v2.0-c",  // string: Full model identifier (HuggingFace model ID or custom name)
      "dataset": "train",                          // string: "train" or "test"
      "seed": 42,                               // int: Seed of used in this run
      "outputs": [                              // list[str]: Labels parsed from language model answers
         "1",
         "5"
      ],
      "output_errors": 5,                       // int: Number of errors in language model output parsing (bad format)
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
