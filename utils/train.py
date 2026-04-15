import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw

from dataloader.dataloader import create_dataloader
from tqdm import tqdm
from PIL import ImageFont


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate(model, data_loader, device, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def _draw_line_chart(draw, left, top, width, height, values_a, values_b, title, y_label, legend_a, legend_b):
    right = left + width
    bottom = top + height
    padding_left = 90
    padding_right = 45
    padding_top = 58
    padding_bottom = 92
    font_size = 16
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 18)
    except Exception:
        title_font = font

    plot_left = left + padding_left
    plot_right = right - padding_right
    plot_top = top + padding_top
    plot_bottom = bottom - padding_bottom

    draw.rectangle([left, top, right, bottom], outline=(210, 210, 210), width=2)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = left + (width - title_width) / 2
    draw.text((title_x, top + 10), title, fill=(20, 20, 20), font=title_font)

    min_value = min(min(values_a), min(values_b))
    max_value = max(max(values_a), max(values_b))
    if abs(max_value - min_value) < 1e-8:
        max_value = min_value + 1.0

    for step in range(5):
        y = plot_top + (plot_bottom - plot_top) * step / 5
        value = max_value - (max_value - min_value) * step / 5
        draw.line([plot_left, y, plot_right, y], fill=(230, 230, 230), width=1)
        draw.text((left + 4, y - 8), f"{value:.2f}", fill=(80, 80, 80), font=font)

    epochs = len(values_a)
    tick_step = max(1, epochs // 6)
    for idx in range(epochs):
        x = plot_left if epochs == 1 else plot_left + (plot_right - plot_left) * idx / (epochs - 1)
        draw.line([x, plot_top, x, plot_bottom], fill=(242, 242, 242), width=1)
        if idx == 0 or idx == epochs - 1 or idx % tick_step == 0:
            draw.text((x - 7, plot_bottom + 8), str(idx + 1), fill=(80, 80, 80), font=font)

    draw.line([plot_left, plot_bottom, plot_right, plot_bottom], fill=(0, 0, 0), width=2)
    draw.line([plot_left, plot_top, plot_left, plot_bottom], fill=(0, 0, 0), width=2)
    draw.text((plot_right - 58, plot_bottom + 28), "Epoch", fill=(20, 20, 20), font=font)
    draw.text((left + 6, plot_top - 28), y_label, fill=(20, 20, 20), font=font)

    def to_points(values):
        points = []
        for idx, value in enumerate(values):
            x = plot_left if epochs == 1 else plot_left + (plot_right - plot_left) * idx / (epochs - 1)
            ratio = (value - min_value) / (max_value - min_value)
            y = plot_bottom - ratio * (plot_bottom - plot_top)
            points.append((x, y))
        return points

    train_points = to_points(values_a)
    test_points = to_points(values_b)
    if len(train_points) > 1:
        draw.line(train_points, fill=(34, 139, 230), width=3)
        draw.line(test_points, fill=(230, 99, 61), width=3)

    for point in train_points:
        draw.ellipse([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill=(34, 139, 230))
    for point in test_points:
        draw.ellipse([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill=(230, 99, 61))

    legend_y = top + 14
    legend_x = right - 165
    draw.line([legend_x, legend_y + 10, legend_x + 30, legend_y + 10], fill=(34, 139, 230), width=3)
    draw.text((legend_x + 36, legend_y), legend_a, fill=(20, 20, 20), font=font)

    second_legend_y = legend_y + 30
    draw.line([legend_x, second_legend_y + 10, legend_x + 30, second_legend_y + 10], fill=(230, 99, 61), width=3)
    draw.text((legend_x + 36, second_legend_y), legend_b, fill=(20, 20, 20), font=font)


def plot_history(history, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    image = Image.new("RGB", (1760, 820), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    _draw_line_chart(
        draw=draw,
        left=40,
        top=40,
        width=760,
        height=660,
        values_a=history["train_acc"],
        values_b=history["test_acc"],
        title="Training and Testing Accuracy",
        y_label="Accuracy (%)",
        legend_a="Train Acc",
        legend_b="Test Acc",
    )
    _draw_line_chart(
        draw=draw,
        left=900,
        top=40,
        width=820,
        height=660,
        values_a=history["train_loss"],
        values_b=history["test_loss"],
        title="Training and Testing Loss",
        y_label="Loss",
        legend_a="Train Loss",
        legend_b="Test Loss",
    )

    plot_path = save_path.replace(".pth", "_curves.png")
    image.save(plot_path)


def build_train_loader(data_root, batch_size=64, num_workers=0, pin_memory=None, drop_last=False, image_size=224, pretrained=False):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    config = {
        "data_root": data_root,
        "batch_size": batch_size,
        "shuffle_train": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "image_size": image_size,
        "pretrained": pretrained,
    }
    return create_dataloader(config, mode="train")


def train(
    model,
    train_loader,
    test_loader,
    device,
    epochs,
    learning_rate,
    save_path="checkpoints/model.pth",
    weight_decay=1e-4,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("Task: classification")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("Learning rate scheduler: cosine")

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        test_loss, test_acc = _evaluate(model, test_loader, device, criterion)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(accuracy)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        if test_acc >= best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)

        tqdm.write(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )
        plot_history(history, save_path)

    print(f"Training done. Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    print("Use main.py to run training with custom arguments")
