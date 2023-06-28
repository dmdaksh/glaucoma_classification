import argparse
import os
import sys

import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, ConcatDataset
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder

from src.dataloaders import get_dataloader
from src.losses import losses
from src.model import models
from src.optim import optimizers
from src.trainer import Trainer

class MAETrainer(Trainer):
    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            loss_fn: nn.Module,
            is_distributed_training: bool = False,
            num_classes: int = 2,
            cuda_device: int = 0
    ):
        super().__init__(model, optimizer, loss_fn, is_distributed_training, num_classes, cuda_device)

    def pretrain(
            self,
            train_loader: DataLoader,
            epochs: int = 25,
            print_every: int = 1,
    ):
        self.model.train()

        scaler = GradScaler()
        for epoch in range(epochs):
            for i, (images, _) in enumerate(train_loader):
                images = images.to(self.cuda_device)

                self.optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    masked_patches_pred, masked_patches_true = self.model(images)
                    loss = self.loss_fn(masked_patches_pred, masked_patches_true)
                # loss.backward()
                scaler.scale(loss).backward()
                # self.optimizer.step()
                scaler.step(optimizer)
                scaler.update()

            if (i + 1) % print_every == 0:
                    print(
                        f"Epoch: {epoch + 1}/{epochs}, Step: {i + 1}/{len(train_loader)}, Loss: {loss.item():.7f}"
                    )
    
    def test_pretrain(
              self,
              data_loader: DataLoader,
    ):
        model_name = self.model.__class__.__name__
        if not os.path.exists(f'images/{model_name}'):
            os.makedirs(f'images/{model_name}')
        self.model.eval()
        for j, (images, _) in enumerate(data_loader):
            images = images.to(self.cuda_device)
            masked_patches_pred, masked_patches_true, x_patches, x_patches_masked, x_patches_pred = self.model(images, test_mode=True)


            for i in range(x_patches.shape[0]):
                true_image = self.model.combine_patches(x_patches[i])
                true_image_masked = self.model.combine_patches(x_patches_masked[i])
                pred_image = self.model.combine_patches(x_patches_pred[i])

                # grid = make_grid([true_image, true_image_masked, pred_image], nrow=3)
                grid = pred_image
                grid = ToPILImage(mode='RGB')(grid)
                grid.save(f'images/{model_name}/pred_gen_{j*32+i}.png')
                i += 1

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=models.keys(), default='mae')
    parser.add_argument("data_csv_path", type=str, default="compiled_file_addr/REFUGE1_crop.csv")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=optimizers.keys()
    )
    parser.add_argument(
        "--loss", type=str, default="mse", choices=losses.keys()
    )
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    print(args)
    model_config = {
         'spatial_dim': 256,
         'n_channels': 3,
         'mask_ratio': 0.75,
         'patch_size': 16
    }
    model = models[args.model](**model_config).cuda()
    # model = models[args.model](model_config, pretrained_weights_path="checkpoints/mae.pth", n_classes=2).cuda()
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # model = nn.DataParallel(model)

    label_encoder = LabelEncoder()

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
        pin_memory=True,
        data_csv=args.data_csv_path,
        transforms=train_transform,
        label_encoder=label_encoder,
        combine_refuge=False
    )
    val_dl = get_dataloader(
        dataset_name="refuge1_crop",
        split="val",
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        data_csv=args.data_csv_path,
        transforms=test_transform,
        label_encoder=label_encoder,
    )
    test_dl = get_dataloader(
        dataset_name="refuge1_crop",
        split="test",
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        data_csv=args.data_csv_path,
        transforms=test_transform,
        label_encoder=label_encoder,
    )

    optimizer = optimizers[args.optimizer](model.parameters(), lr=args.lr)

    if args.loss == "cross_entropy":
        class_imbalance = torch.zeros(2)
        for _, labels in train_dl:
            class_imbalance += torch.bincount(labels, minlength=2)
        class_imbalance = class_imbalance / class_imbalance.sum()
        print(f"class imbalance: {class_imbalance}")
        class_weights = 1-class_imbalance
        loss = losses[args.loss](weight=class_weights.to(device=0))
        # loss = losses[args.loss]()
    else:
        loss = losses[args.loss]()

    trainer = MAETrainer(model, optimizer, loss)

    # print("=> Pretraining")
    # trainer.pretrain(train_dl, epochs=args.epochs)

    # print("=> Saving checkpoint")
    # trainer.save_checkpoint(os.path.join("checkpoints", f"{args.model}.pth"))

    print("=> Loading checkpoint")
    trainer.load_checkpoint(os.path.join("checkpoints", f"{args.model}.pth"))

    print("=> Test model")
    trainer.test_pretrain(test_dl)

    # print("=> Linear Probing")
    # trainer.train(train_dl, val_dl, epochs=args.epochs, print_every=3)

    # print("=> Testing")
    # trainer.test(test_dl)

    # print("=> Saving predictions")
    # predictions = trainer.predict(test_dl)
    # trainer.save_predictions(f"predictions/{args.model}", predictions, label_encoder=label_encoder)
