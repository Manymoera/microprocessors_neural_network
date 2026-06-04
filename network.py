import os
import copy
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import googlenet, GoogLeNet_Weights
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs.logits
        else:
            outputs = outputs
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    return epoch_loss, epoch_acc


def build_model(num_classes=2, freeze_backbone=False):
    weights = GoogLeNet_Weights.IMAGENET1K_V1

    model = googlenet(
        weights=weights,
        aux_logits=True
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def export_torchscript(model, export_path, device):
    model.eval()
    model.to(device)

    example_input = torch.randn(1, 3, 224, 224).to(device)

    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(export_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--export_dir", type=str, default="exports")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transforms
    )

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    model = build_model(
        num_classes=len(class_names),
        freeze_backbone=args.freeze_backbone
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    best_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

            checkpoint_path = os.path.join(
                args.export_dir,
                "best_googlenet_face.pth"
            )

            torch.save({
                "model_state_dict": best_model_weights,
                "class_names": class_names,
                "val_acc": best_acc
            }, checkpoint_path)

            print(f"Best model saved: {checkpoint_path}")

    model.load_state_dict(best_model_weights)

    final_checkpoint_path = os.path.join(
        args.export_dir,
        "final_googlenet_face.pth"
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "val_acc": best_acc
    }, final_checkpoint_path)

    print(f"\nFinal model saved: {final_checkpoint_path}")

    torchscript_path = os.path.join(
        args.export_dir,
        "googlenet_face_torchscript.pt"
    )

    export_torchscript(
        model=model,
        export_path=torchscript_path,
        device=device
    )

    print(f"TorchScript model exported: {torchscript_path}")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
