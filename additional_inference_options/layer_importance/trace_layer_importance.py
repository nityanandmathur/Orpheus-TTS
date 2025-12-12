import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orpheus_tts import AcousticCausalTracer, plot_layer_importance


def parse_codec_ids(ids_str, device):
    if not ids_str:
        return None
    try:
        parts = (part.strip() for part in ids_str.split(","))
        values = [int(part) for part in parts if part]
    except ValueError as parse_error:
        raise ValueError(
            f"Invalid --target-codec-ids value '{ids_str}'. Expect comma-separated integers."
        ) from parse_error
    if not values:
        return None
    return torch.tensor(values, device=device).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Trace layer importance for a target word.")
    parser.add_argument("--model-id", required=True, help="HF repo id or local path for the Orpheus checkpoint.")
    parser.add_argument("--text", required=True, help="Prompt text (include voice tag as needed).")
    parser.add_argument("--word", required=True, help="Target word to trace.")
    parser.add_argument("--target-codec-ids", help="Comma-separated codec ids for the expected audio tokens.")
    parser.add_argument("--use-argmax-target", action="store_true", help="Use greedy tokens from a clean pass as targets.")
    parser.add_argument("--noise-level", type=float, default=0.1, help="Noise magnitude added to the target token embedding.")
    parser.add_argument("--save-path", default="layer_importance.png", help="Where to save the plot.")
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.eval()

    tracer = AcousticCausalTracer(model, tokenizer, device=device)

    try:
        target_codec_ids = parse_codec_ids(args.target_codec_ids, device)
    except ValueError as exc:
        parser.error(str(exc))

    if target_codec_ids is None and args.use_argmax_target:
        encoded = tokenizer(args.text, return_tensors="pt").to(device)
        with torch.no_grad():
            clean_out = model(**encoded)
        target_codec_ids = clean_out.logits.argmax(dim=-1)

    scores, target_token_idx = tracer.trace_from_text(
        args.text,
        args.word,
        target_codec_ids=target_codec_ids,
        noise_level=args.noise_level,
    )

    plot_layer_importance(scores, save_path=args.save_path, show=args.show)
    print(
        f"Saved layer importance plot to {args.save_path} (target token index: {target_token_idx})"
    )


if __name__ == "__main__":
    main()
