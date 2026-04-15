import argparse
import os
import torch
from models import Model
from utils.test import build_test_loader, test
from utils.train import build_train_loader, set_seed, train



def parse_args():
    parser = argparse.ArgumentParser(description="SVHN Format 2 classification with built-in PyTorch models")

    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-name", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    save_path = os.path.join(args.save_dir, "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
    ).to(device)

    train_loader = build_train_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        image_size=args.image_size,
        pretrained=args.pretrained,
    )
    test_loader = build_test_loader(
        data_root=args.data_root,
        batch_size=128,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        image_size=args.image_size,
        pretrained=args.pretrained,
    )
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=5e-3,
        save_path=save_path,
        weight_decay=1e-4,
    )
    test(model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
