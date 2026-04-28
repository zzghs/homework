import json
import os
import random

import matplotlib
matplotlib.use("Agg")

import torch
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epochs,
    save_path,
    history_path=None,
):
    history = {"loss": []}
    best_loss = float("inf")
    total_steps = epochs * len(train_loader)

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Train steps per epoch: {len(train_loader)}")

    progress_bar = tqdm(total=total_steps, desc="Training", leave=False)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix(epoch=f"{epoch}/{epochs}", loss=f"{loss.item():.4f}")

        average_loss = running_loss / total_steps
        history["loss"].append(average_loss)

        if average_loss < best_loss:
            best_loss = average_loss
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if history_path is not None:
            os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
            with open(history_path, "w", encoding="utf-8") as history_file:
                json.dump(history, history_file, ensure_ascii=False, indent=2)

    progress_bar.close()

    return history


def plot_training_curve(loss_history, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", markersize=1, linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_generated_poems(poems, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as output_file:
        for index, poem in enumerate(poems, start=1):
            output_file.write(f"Poem {index}\n")
            output_file.write(poem)
            output_file.write("\n\n")
