import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

import argparse

def refuge():
    addr = "datasets/REFUGE/"
    glaucoma_addr = addr + "glaucoma/images/"
    normal_addr = addr + "normal/images/"
    # list of files in glaucoma_addr
    glaucoma_files = os.listdir(glaucoma_addr)
    normal_files = os.listdir(normal_addr)

    if not os.path.exists("compiled_file_addr"):
        os.mkdir("compiled_file_addr")

    with open("compiled_file_addr/REFUGE.csv", "w") as f:
        f.write("Image Path, Label\n")
        for glaucoma_file in glaucoma_files:
            f.write(glaucoma_addr + glaucoma_file + "," + "glaucoma\n")

        for normal_file in normal_files:
            f.write(normal_addr + normal_file + "," + "normal\n")

def refuge1():
    addr = "datasets/REFUGE1/"

    if not os.path.exists("compiled_file_addr"):
        os.mkdir("compiled_file_addr")

    filepaths = {}
    for root, dirs, files in os.walk(addr):
        if len(files) == 0:
            continue

        for f in files:
            if ("Images" in root) and (f.endswith(".jpg")):
                filename = f.split(".")[0]
                if filename not in filepaths:
                    filepaths[filename] = {}
                filepaths[filename]["image"] = os.path.join(root, f)
            elif ("Disc_Cup_Masks" in root) and (f.endswith(".bmp")):
                filename = f.split(".")[0]
                if filename not in filepaths:
                    filepaths[filename] = {}
                filepaths[filename]["mask"] = os.path.join(root, f)

    columns = ["split", "image filepath", "mask filepath", "label"]
    df = pd.DataFrame(columns=columns)

    # assign imagefilepath and maskfilepath to make sure they are in the same order
    for filename, filepath in filepaths.items():
        df = df._append(
            {
                "image filepath": filepath["image"],
                "mask filepath": filepath["mask"],
            },
            ignore_index=True,
        )
    # if filepath has glaucoma in it, then label it as glaucoma
    df["label"] = df["image filepath"].apply(
        lambda x: "normal" if "Non-Glaucoma" in x else "glaucoma"
    )
    df["split"] = df["image filepath"].apply(
        lambda x: "train" if "Train" in x else ("val" if "Val" in x else "test") 
    )

    df.to_csv("compiled_file_addr/REFUGE1.csv", index=False)

    return df


def refuge1_crop():
    addr = 'preprocessed_data/REFUGE1'

    filepaths = {}
    for root, dirs, files in os.walk(addr):
        if len(files) == 0:
            continue

        for f in files:
            if f.endswith('.jpg'):
                filename = f.split('.')[0]
                filepaths[filename] = {}
                filepaths[filename]['image'] = os.path.join(root, f)
        
    columns = ['split', 'crop image filepath', 'label']
    df = pd.DataFrame(columns=columns)

    for filename, filepath in filepaths.items():
        df = df._append(
            {
                'crop image filepath': filepath['image'],
            },
            ignore_index=True,
        )

    df['label'] = df['crop image filepath'].apply(
        lambda x: 'normal' if 'normal' in x else 'glaucoma'
    )
    df['split'] = df['crop image filepath'].apply(
        lambda x: 'train' if 'train' in x else ('val' if 'val' in x else 'test')
    )
    
    df.to_csv('compiled_file_addr/REFUGE1_crop.csv', index=False)

class acrima_refuge_crop():
    refuge1_crop()

    df = pd.read_csv('compiled_file_addr/REFUGE1_crop.csv')
    acrima_addr = 'datasets/ACRIMA/combined'

    filepaths = {}
    for root, dirs, files in os.walk(acrima_addr):
        if len(files) == 0:
            continue

        for f in files:
            if f.endswith('.jpg'):
                filename = f.split('.')[0]
                filepaths[filename] = {}
                filepaths[filename]['image'] = os.path.join(root, f)
        
    columns = ['split', 'crop image filepath', 'label']
    acrima_df = pd.DataFrame(columns=columns)

    for filename, filepath in filepaths.items():
        acrima_df = acrima_df._append(
            {
                'crop image filepath': filepath['image'],
            },
            ignore_index=True,
        )
    acrima_df['label'] = acrima_df['crop image filepath'].apply(
        lambda x: 'glaucoma' if '_g_' in x else 'normal'
    )
    acrima_df['split'] = 'train'
    
    df = pd.concat([df, acrima_df])

    df.to_csv('compiled_file_addr/ACRIMA_REFUGE_crop.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('function', type=str, )

    args = parser.parse_args()

    if args.function == 'refuge':
        refuge()
    elif args.function == 'refuge1':
        refuge1()
    elif args.function == 'refuge1_crop':
        refuge1_crop()
    elif args.function == 'acrima_refuge_crop':
        acrima_refuge_crop()


