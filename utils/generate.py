import torch


PUNCTUATION_RULES = {7: "，", 15: "。", 23: "，", 31: "。"}


def _sample_from_logits(logits, allowed_ids, temperature=0.8, top_k=8):
    masked_logits = logits.clone()
    disallowed = torch.ones_like(masked_logits, dtype=torch.bool)
    disallowed[allowed_ids] = False
    masked_logits[disallowed] = float("-inf")
    masked_logits = masked_logits / max(temperature, 1e-5)

    finite_mask = torch.isfinite(masked_logits)
    filtered_ids = torch.nonzero(finite_mask, as_tuple=False).squeeze(-1)
    filtered_logits = masked_logits[filtered_ids]

    if top_k is not None and 0 < top_k < filtered_logits.numel():
        top_values, top_indices = torch.topk(filtered_logits, top_k)
        top_ids = filtered_ids[top_indices]
        probabilities = torch.softmax(top_values, dim=-1)
        sampled_index = torch.multinomial(probabilities, num_samples=1).item()
        return top_ids[sampled_index].item()

    probabilities = torch.softmax(filtered_logits, dim=-1)
    sampled_index = torch.multinomial(probabilities, num_samples=1).item()
    return filtered_ids[sampled_index].item()


def format_qiyan_jueju(poem_text):
    return f"{poem_text[:16]}\n{poem_text[16:]}"


def generate_poems(
    model,
    vocabulary,
    device,
    prefix="明月",
    count=2,
    temperature=0.8,
    top_k=8,
):
    if len(prefix) > 7:
        raise ValueError("The prefix must not exceed seven characters.")

    punctuation_ids = {vocabulary.stoi[char] for char in {"，", "。"} if char in vocabulary.stoi}
    special_ids = {vocabulary.pad_id, vocabulary.bos_id, vocabulary.eos_id, vocabulary.unk_id}
    content_ids = [
        token_id
        for token_id, token in enumerate(vocabulary.itos)
        if token_id not in special_ids and token_id not in punctuation_ids
    ]

    model.eval()
    results = []
    seen = set()

    while len(results) < count:
        generated_ids = [vocabulary.bos_id]
        generated_text = ""

        for position in range(32):
            if position < len(prefix):
                next_token_id = vocabulary.stoi[prefix[position]]
            else:
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
                logits, _ = model(input_tensor)
                next_logits = logits[0, -1]
                if position in PUNCTUATION_RULES:
                    next_token_id = vocabulary.stoi[PUNCTUATION_RULES[position]]
                else:
                    next_token_id = _sample_from_logits(
                        logits=next_logits,
                        allowed_ids=content_ids,
                        temperature=temperature,
                        top_k=top_k,
                    )

            generated_ids.append(next_token_id)
            generated_text += vocabulary.itos[next_token_id]

        if generated_text not in seen:
            seen.add(generated_text)
            results.append(format_qiyan_jueju(generated_text))

    return results
