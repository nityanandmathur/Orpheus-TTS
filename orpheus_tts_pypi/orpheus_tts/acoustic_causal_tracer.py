import torch
import torch.nn.functional as F


class AcousticCausalTracer:
    """
    Implements a causal tracing routine over decoder layers to measure the
    contribution of each layer to the pronunciation (codec token prediction)
    of a specific word. Based on the reference snippet provided in the issue.
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

        self.core = getattr(model, "model", model)
        self.layers = getattr(self.core, "layers", None)
        if self.layers is None:
            raise ValueError("Model must expose decoder layers via model.layers")

        self.embed_tokens = getattr(self.core, "embed_tokens", None)
        if self.embed_tokens is None:
            get_embeddings = getattr(model, "get_input_embeddings", None)
            self.embed_tokens = get_embeddings() if callable(get_embeddings) else None
        if self.embed_tokens is None:
            raise ValueError(
                "Model must expose token embeddings via model.embed_tokens or get_input_embeddings()."
            )

        self.num_layers = len(self.layers)

    def _extract_hidden(self, output):
        """
        Extract the hidden state tensor from a decoder layer output, accounting
        for ModelOutput or tuple variants.
        """
        if isinstance(output, tuple):
            return output[0]

        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state

        if hasattr(output, "hidden_states"):
            hidden = output.hidden_states
            return hidden[-1] if isinstance(hidden, (list, tuple)) else hidden

        return output

    def _replace_hidden(self, output, new_hidden):
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]

        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = new_hidden
            return output

        if hasattr(output, "hidden_states") and not hasattr(output, "last_hidden_state"):
            output.hidden_states = new_hidden
            return output

        return new_hidden

    def _normalize_target_ids(self, target_codec_ids, logits):
        if target_codec_ids is None:
            target_codec_ids = logits.argmax(dim=-1)

        if isinstance(target_codec_ids, torch.Tensor):
            target_ids = target_codec_ids.to(logits.device)
        else:
            target_ids = torch.tensor(target_codec_ids, device=logits.device)

        if target_ids.ndim == 1:
            target_ids = target_ids.unsqueeze(0)

        return target_ids

    def _get_prob(self, outputs, target_ids):
        logits = outputs.logits
        target_ids = self._normalize_target_ids(target_ids, logits)

        seq_len = min(target_ids.shape[1], logits.shape[1])
        target_ids = target_ids[:, -seq_len:]
        logits = logits[:, -seq_len:, :]

        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return target_log_probs.mean()

    def _get_token_index(self, input_ids, target_word):
        word_token_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
        if len(word_token_ids) == 0:
            raise ValueError("Target word produced no tokens.")

        input_list = input_ids.tolist()
        for start in range(len(input_list) - len(word_token_ids) + 1):
            if input_list[start : start + len(word_token_ids)] == word_token_ids:
                return start

        raise ValueError(f"Target word '{target_word}' not found in the provided text.")

    def get_activations(self, inputs, noun_token_idx):
        activations = {}

        def get_hook(layer_idx):
            def hook(module, _input, output):
                hidden = self._extract_hidden(output)
                activations[layer_idx] = hidden[:, noun_token_idx, :].detach().clone()

            return hook

        handles = [layer.register_forward_hook(get_hook(i)) for i, layer in enumerate(self.layers)]

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        return activations

    def trace(
        self,
        inputs,
        noun_idx,
        target_codec_ids=None,
        noise_level=0.1,
    ):
        clean_activations = self.get_activations(inputs, noun_idx)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        clean_embeddings = self.embed_tokens(input_ids)
        noise = torch.randn_like(clean_embeddings[:, noun_idx, :]) * noise_level
        corrupted_embeddings = clean_embeddings.clone()
        corrupted_embeddings[:, noun_idx, :] += noise

        forward_kwargs = {"inputs_embeds": corrupted_embeddings}
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            corrupted_out = self.model(**forward_kwargs)
            base_prob = self._get_prob(corrupted_out, target_codec_ids)

        scores = []

        def make_patch_hook(idx):
            def patch_hook(module, _input, output):
                hidden = self._extract_hidden(output)
                patched_hidden = hidden.clone()
                patched_hidden[:, noun_idx, :] = clean_activations[idx]
                return self._replace_hidden(output, patched_hidden)

            return patch_hook

        for layer_idx in range(self.num_layers):
            handle = self.layers[layer_idx].register_forward_hook(make_patch_hook(layer_idx))

            try:
                with torch.no_grad():
                    patched_out = self.model(**forward_kwargs)
                    restored_prob = self._get_prob(patched_out, target_codec_ids)
            finally:
                handle.remove()

            scores.append((restored_prob - base_prob).item())

        return scores

    def trace_from_text(
        self,
        text,
        target_word,
        target_codec_ids=None,
        noise_level=0.1,
    ):
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        noun_idx = self._get_token_index(encoded["input_ids"][0], target_word)
        return self.trace(encoded, noun_idx, target_codec_ids, noise_level), noun_idx


def plot_layer_importance(scores, save_path=None, show=False):
    # Lazy import keeps matplotlib optional for non-plotting consumers. Install it when using this helper.
    import matplotlib.pyplot as plt

    layers = list(range(len(scores)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, scores, marker="o")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Indirect effect on log-prob")
    ax.set_title("Layer importance for pronunciation")
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
