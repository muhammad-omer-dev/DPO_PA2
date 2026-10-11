"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def build_sft_batch(tokenizer, prompts, responses, max_length):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    eos_id = tokenizer.eos_token_id

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        if eos_id is not None:
            response_ids.append(eos_id)

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids.copy()

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        attention_mask = [1] * len(input_ids)

        while len(input_ids) < max_length:
            input_ids.append(pad_id)
            labels.append(-100)
            attention_mask.append(0)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
    }


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)

    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits.float()
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    log_probs = F.log_softmax(shifted_logits, dim=-1)

    safe_labels = shifted_labels.clone()
    safe_labels[safe_labels == -100] = 0

    token_logps = log_probs.gather(
        2,
        safe_labels.unsqueeze(2),
    ).squeeze(2)

    response_mask = shifted_labels != -100
    token_logps = token_logps * response_mask

    return token_logps.sum(dim=1)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)

    response_logps = compute_response_logprobs(
        model,
        tokenizer,
        prompts,
        responses,
        max_length,
    )

    labels = batch["labels"][:, 1:]
    response_token_count = (labels != -100).sum()
    response_token_count = response_token_count.clamp_min(1)

    loss = -response_logps.sum() / response_token_count.to(response_logps.device)
    return loss


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
):
    chosen_batch = build_sft_batch(
        tokenizer,
        prompts,
        chosen_responses,
        max_length,
    )

    rejected_batch = build_sft_batch(
        tokenizer,
        prompts,
        rejected_responses,
        max_length,
    )

    return {
        "chosen": chosen_batch,
        "rejected": rejected_batch,
    }


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps

    dpo_logits = beta * (policy_gap - reference_gap)
    loss = -F.logsigmoid(dpo_logits).mean()

    return loss


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    policy_chosen = compute_response_logprobs(
        policy_model,
        tokenizer,
        prompts,
        chosen_responses,
        max_length,
    )

    policy_rejected = compute_response_logprobs(
        policy_model,
        tokenizer,
        prompts,
        rejected_responses,
        max_length,
    )

    with torch.no_grad():
        reference_chosen = compute_response_logprobs(
            reference_model,
            tokenizer,
            prompts,
            chosen_responses,
            max_length,
        )

        reference_rejected = compute_response_logprobs(
            reference_model,
            tokenizer,
            prompts,
            rejected_responses,
            max_length,
        )

    reference_chosen = reference_chosen.to(policy_chosen.device)
    reference_rejected = reference_rejected.to(policy_chosen.device)

    return compute_dpo_loss_from_logps(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta,
    )

