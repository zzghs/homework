import argparse
import os

import torch
from torch import nn, optim

from dataloader import PoetryDataset, PoetryVocabulary, create_dataloader, load_qiyan_jueju_corpus
from models import PoetryGenerator
from utils import generate_poems, plot_training_curve, set_seed, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate seven-character quatrains with a built-in PyTorch RNN/LSTM.")
    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", type=str, default="lstm", choices=["lstm", "rnn"])
    parser.add_argument("--generate-count", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    poems = load_qiyan_jueju_corpus(args.data_root)
    vocabulary = PoetryVocabulary(poems, min_freq=2)
    dataset = PoetryDataset(poems, vocabulary)
    train_loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    try:
        torch.zeros(1, device="cuda")
        device = torch.device("cuda")
    except Exception as error:
        print(f"CUDA is unavailable at runtime, fallback to CPU: {error}")
        device = torch.device("cpu")

    model = PoetryGenerator(
        vocab_size=len(vocabulary),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=args.dropout,
        model_type=args.model_type,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "best_model.pth")
    curve_path = os.path.join(args.save_dir, "training_loss_curve.png")

    print(f"Corpus size: {len(poems)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Model type: {args.model_type.upper()}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        save_path=checkpoint_path,
        history_path=None,
    )

    plot_training_curve(history["loss"], curve_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    poems = generate_poems(
        model=model,
        vocabulary=vocabulary,
        device=device,
        prefix="\u660e\u6708",
        count=args.generate_count,
        temperature=0.8,
        top_k=8,
    )
    # No longer save generated poems to file, only print to console

    print("Generated poems:")
    for index, poem in enumerate(poems, start=1):
        print(f"Poem {index}")
        print(poem)


if __name__ == "__main__":
    main()
