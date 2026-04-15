import argparse
import os

import torch
import torch.nn as nn

from dataloader.dataloader import create_dataloader
from models import Model


def build_test_loader(data_root, batch_size=64, num_workers=0, pin_memory=None, image_size=224, pretrained=False):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    config = {
        "data_root": data_root,
        "batch_size": batch_size,
        "test_batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "image_size": image_size,
        "pretrained": pretrained,
    }
    return create_dataloader(config, mode="test")


def test(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    print("Task: classification")
    print(f"Device: {device}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("-" * 60)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("-" * 60)
    print("Testing done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation using a saved SVHN model")
    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--save-path", type=str, default=os.path.join("results", "best_model.pth"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34"],
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_path):
        raise FileNotFoundError(f"Model file not found: {args.save_path}")

    model = Model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
    ).to(device)
    model.load_state_dict(torch.load(args.save_path, map_location=device))

    test_loader = build_test_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        image_size=args.image_size,
        pretrained=args.pretrained,
    )
    test(model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()

