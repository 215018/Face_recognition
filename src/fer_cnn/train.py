# Trains the Proposed_Model_2 LM-CNN using early-fused grayscale image and landmark heatmap inputs.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch import nn
from sklearn.metrics import classification_report, precision_recall_fscore_support

from fer_cnn.config import TrainConfig
from fer_cnn.dataset import build_loaders
from fer_cnn.model import ProposedModel2LMCNN


def compute_class_weights(
    dataset,
    num_classes: int,
    device: torch.device,
    power: float,
):
    # Weight rare classes higher so imbalance does not dominate the loss.
    labels = torch.tensor([dataset.dataset.targets[index] for index in dataset.indices])
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    weights = weights.pow(power)
    weights = weights / weights.mean()
    return weights.to(device)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    # Train model for one full pass through the training data.
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

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def evaluate(model, loader, loss_fn, device):
    # Evaluate model without updating weights.
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
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy, all_labels, all_predictions


def print_metrics(name: str, labels: list[int], predictions: list[int], class_names):
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

    print(f"{name} Macro Precision: {macro_precision:.4f}")
    print(f"{name} Macro Recall: {macro_recall:.4f}")
    print(f"{name} Macro F1: {macro_f1:.4f}")
    print(f"{name} Weighted Precision: {weighted_precision:.4f}")
    print(f"{name} Weighted Recall: {weighted_recall:.4f}")
    print(f"{name} Weighted F1: {weighted_f1:.4f}")
    print(
        classification_report(
            labels,
            predictions,
            target_names=class_names,
            zero_division=0,
        )
    )


def main():
    # Load training configuration.
    config = TrainConfig()

    # Create output folders.
    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    # Use GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build dataset loaders.
    train_loader, val_loader, test_loader, class_names = build_loaders(config)
    print("Classes:", class_names)

    # Create Proposed_Model_2 LM-CNN.
    model = ProposedModel2LMCNN(num_classes=len(class_names)).to(device)

    class_weights = compute_class_weights(
        train_loader.dataset,
        num_classes=len(class_names),
        device=device,
        power=config.class_weight_power,
    )
    print("Class weights:", class_weights.detach().cpu().tolist())

    # Weighted loss helps the model learn underrepresented emotions.
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )

    # Adam optimizer uses the Proposed_Model_2 training setup.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )

    best_val_accuracy = -1.0
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

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Save model if validation accuracy improves.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "image_size": config.image_size,
                    "best_val_accuracy": best_val_accuracy,
                    "class_weights": class_weights.detach().cpu(),
                },
                "checkpoints/best_lm_cnn.pt",
            )

            print("Saved best model.")
        else:
            epochs_without_improvement += 1

        scheduler.step(val_accuracy)

        # Early stopping to reduce overfitting.
        if epochs_without_improvement >= config.patience:
            print("Early stopping triggered.")
            break

    # Test final best model.
    checkpoint = torch.load("checkpoints/best_lm_cnn.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_accuracy, test_labels, test_predictions = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print_metrics("Test", test_labels, test_predictions, class_names)


if __name__ == "__main__":
    main()
