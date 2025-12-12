## Layer importance tracing for pronunciation

This example shows how to run the `AcousticCausalTracer` (added to the package) and
plot a simple graph of how much each decoder layer contributes to the pronunciation
of a target word.

### Setup

Install the optional dependencies needed for tracing and plotting:

```bash
pip install torch transformers matplotlib
```

### Run

```bash
python additional_inference_options/layer_importance/trace_layer_importance.py \
  --model-id canopylabs/orpheus-tts-0.1-finetune-prod \
  --text "zoe: The quick brown fox jumps over the lazy dog." \
  --word "fox" \
  --use-argmax-target \
  --save-path layer_importance.png
```

Arguments:
- `--model-id`: Hugging Face repo or local path for an Orpheus model checkpoint.
- `--text`: Prompt text (include a voice prefix if required by your checkpoint).
- `--word`: Target word whose pronunciation you want to trace.
- `--target-codec-ids`: Optional comma-separated codec ids (e.g., `12,34,56`). If
  omitted, you can pass `--use-argmax-target` to use the model's greedy prediction
  from a clean pass.
- `--noise-level`: Magnitude of noise added to the target token embedding (default
  0.1).
- `--save-path`: Where to write the plotted PNG.
- `--show`: Also display the plot window (if your environment supports it).

The script prints the token index for the target word and saves a graph with the
indirect effect (improvement over corrupted logits) for every layer. Higher values
indicate layers that were more important for restoring the word's pronunciation.
