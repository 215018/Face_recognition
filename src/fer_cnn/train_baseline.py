# Trains the Proposed_Model_2 baseline using only grayscale FER-2013 images.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch import nn

from fer_cnn.baseline_dataset import build_baseline_loaders
from fer_cnn.baseline_model import ProposedModel2Baseline
from fer_cnn.config import TrainConfig


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return total_loss / total, correct / total, all_labels, all_predictions


def main():
    config = TrainConfig()

    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = build_baseline_loaders(config)
    print("Classes:", class_names)

    model = ProposedModel2Baseline(num_classes=len(class_names)).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=getattr(config, "lr_scheduler_patience", 3),
        min_lr=getattr(config, "min_learning_rate", 0.00001),
    )

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        val_loss, val_accuracy, _, _ = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        scheduler.step(val_accuracy)

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "image_size": config.image_size,
                    "best_val_accuracy": best_val_accuracy,
                },
                "checkpoints/best_baseline_model2.pt",
            )

            print("Saved best baseline model.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print("Early stopping triggered.")
            break

    checkpoint = torch.load("checkpoints/best_baseline_model2.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_accuracy, labels, predictions = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro Precision: {macro_precision:.4f}")
    print(f"Test Macro Recall: {macro_recall:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"Test Weighted Precision: {weighted_precision:.4f}")
    print(f"Test Weighted Recall: {weighted_recall:.4f}")
    print(f"Test Weighted F1: {weighted_f1:.4f}")

    print(
        classification_report(
            labels,
            predictions,
            target_names=class_names,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()