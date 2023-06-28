import os
from typing import Any
import numpy as np
import pandas as pd
import torch
from src.utils import refuge1
from PIL import Image


# class CustomCrop:
#     def __init__(self, pad):
#         self.pad = pad

#     def __call__(self, img, mask):
#         bbox = Image.fromarray(mask).getbbox()
#         bbox = (
#             bbox[0] - self.pad,
#             bbox[1] - self.pad,
#             bbox[2] + self.pad,
#             bbox[3] + self.pad,
#         )

#         img = img.crop(bbox)
#         return img


def refuge1_crop(pad = 32):
    # transforms = torchvision.transforms.Compose(
    #     [
    #         CustomCrop(16),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Resize((512, 512)),
    #     ]
    # )
    # print("=> Cropping Refuge1 dataset to include only disc and cup")
    # print("=> This will take a while")
    # print("=> Loading dataset")
    # train_ds = Refuge1Dataset(split="train", transform=transforms, need_mask=True)
    # # val_ds = Refuge1Dataset(split="val", transform=transforms, need_mask=True)
    # # test_ds = Refuge1Dataset(split="test", transform=transforms, need_mask=True)
    # print("=> Done loading dataset")

    # print("=> Saving cropped dataset")
    # torch.save(train_ds, "datasets/REFUGE1/train_cropped.pt")
    # print("=> Done saving train dataset")

    print("=> Reading REFUGE1.csv")
    df = pd.read_csv("compiled_file_addr/REFUGE1.csv")


    if not os.path.exists('preprocessed_data/REFUGE1'):
        os.makedirs('preprocessed_data/REFUGE1' )
    
    for i, row in df.iterrows():
        print(f"=> Processing image {i+1}/{len(df)}")
        img = Image.open(row["image filepath"])
        mask = Image.open(row["mask filepath"])
        mask = Image.eval(mask, lambda x: 0 if x == 255 else 1)
        bbox = mask.getbbox()
        bbox = (
            bbox[0] - pad,
            bbox[1] - pad,
            bbox[2] + pad,
            bbox[3] + pad,
        )
        img_crop = img.crop(bbox)

        try:
            img_crop.save(f"preprocessed_data/REFUGE1/{row['split']}/{row['label']}/{row['image filepath'].split('/')[-1]}")
        except FileNotFoundError:
            os.makedirs(f"preprocessed_data/REFUGE1/{row['split']}/{row['label']}")
            img_crop.save(f"preprocessed_data/REFUGE1/{row['split']}/{row['label']}/{row['image filepath'].split('/')[-1]}")


if __name__ == "__main__":
    refuge1_crop()