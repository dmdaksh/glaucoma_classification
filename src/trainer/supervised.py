import argparse
import os
import sys

import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder

from src.dataloaders import get_dataloader
from src.losses import losses
from src.model import models
from src.optim import optimizers
from src.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=models.keys())
    parser.add_argument("data_csv_path", type=str)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=optimizers.keys()
    )
    parser.add_argument(
        "--loss", type=str, default="cross_entropy", choices=losses.keys()
    )
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    print(args)
    model = models[args.model](num_classes=2).cuda()
    # model = nn.DataParallel(model)

    label_encoder = LabelEncoder()
    print('here: ', id(label_encoder))

    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    print("=> Getting dataloaders")
    train_dl = get_dataloader(
        dataset_name="refuge1_crop",
        split="train",
        batch_size=args.batch_size,
        num_workers=4,
        data_csv=args.data_csv_path,
        transforms=train_transform,
        label_encoder=label_encoder,
    )
    val_dl = get_dataloader(
        dataset_name="refuge1_crop",
        split="val",
        batch_size=args.batch_size,
        num_workers=4,
        data_csv=args.data_csv_path,
        transforms=test_transform,
        label_encoder=label_encoder,
    )
    test_dl = get_dataloader(
        dataset_name="refuge1_crop",
        split="test",
        batch_size=args.batch_size,
        num_workers=4,
        data_csv=args.data_csv_path,
        transforms=test_transform,
        label_encoder=label_encoder,
    )

    # get class imbalance from train_dl
    class_imbalance = torch.zeros(2)
    for _, labels in train_dl:
        class_imbalance += torch.bincount(labels, minlength=2)
    class_imbalance = class_imbalance / class_imbalance.sum()
    print(class_imbalance)
    class_weights = 1 - class_imbalance
    optimizer = optimizers[args.optimizer](model.parameters(), lr=args.lr)
    loss = losses[args.loss](weight=class_weights.to(device=0))

    trainer = Trainer(model, optimizer, loss)

    print("=> Training")
    trainer.train(train_dl, val_dl, epochs=args.epochs, print_every=5)

    print("=> Saving checkpoint")
    trainer.save_checkpoint(os.path.join("checkpoints", f"{args.model}.pth"))

    print("=> Testing")
    trainer.test(test_dl)

    print("=> Saving predictions")
    predictions = trainer.predict(test_dl)
    trainer.save_predictions(f"predictions/{args.model}", predictions, label_encoder=label_encoder)
