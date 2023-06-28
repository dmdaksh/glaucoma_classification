import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        is_distributed_training: bool = False,
        num_classes: int = 2,
        cuda_device: int = 0,
    ):
        # cuda device list
        self.devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
        self.cuda_device = cuda_device
        self.is_distributed_training = is_distributed_training
        if self.is_distributed_training:
            self.model = nn.parallel.DistributedDataParallel(
                model, device_ids=self.devices, output_device=self.devices[0]
            )
        else:
            self.model = model.to(self.cuda_device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_classes = num_classes

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 1,
        print_every: int = 10,
    ):
        self.model.train()
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.cuda_device)
                labels = labels.to(self.cuda_device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if (i + 1) % print_every == 0:
                    print(
                        f"Epoch: {epoch + 1}/{epochs}, Step: {i + 1}/{len(train_loader)}, Training Loss: {loss.item():.5f}"
                    )

            self._validate(val_loader)

    def _validate(self, val_loader: DataLoader):
        self.model.eval()
        predicted_ls = []
        labels_ls = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.cuda_device)
                labels = labels.to(self.cuda_device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                predicted_ls.extend(predicted.detach().cpu().numpy())
                labels_ls.extend(labels.detach().cpu().numpy())

        print('Validation - ', end='')
        self.evaluate_metrics(labels_ls, predicted_ls)
        self.model.train()
    
    def evaluate_metrics(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auroc = auc(fpr, tpr)
        auprc = auc(recall, precision)
        print(f'auroc: {auroc:.5f}, auprc: {auprc:.5f}')


    def save_checkpoint(self, path: str):
        # check if all dirs in path exist
        dirs = path.split("/")
        for i in range(1, len(dirs)):
            if not os.path.exists("/".join(dirs[:i])):
                os.mkdir("/".join(dirs[:i]))

        torch.save(self.model.state_dict(), path)
    
    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def test(self, test_loader: DataLoader):
        self._validate(test_loader)

    def predict(self, test_loader: DataLoader):
        self.model.eval()
        result = {'image': [], 'label': [], 'prediction': []}
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.cuda_device)
                labels = labels.to(self.cuda_device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                result['image'].extend(images.detach().cpu().numpy()*255.0)
                result['label'].extend(labels.detach().cpu().numpy())
                result['prediction'].extend(predicted.detach().cpu().numpy())
        
        self.evaluate_metrics(result['label'], result['prediction'])
        print(f"Confusion Matrix: {confusion_matrix(result['label'], result['prediction'])}")
        self.model.train()
        return result
    
    def save_predictions(self, path: str, result: dict, **kwargs):
        print('here: ', id(kwargs.get('label_encoder')))
        print(f'len: {len(result["image"])}, shape of image: {result["image"][0].shape}')
        # check if all dirs in path exist
        if not os.path.exists(path):
            os.makedirs(path)

        
        # inverse transform label encoding of true and predicted labels
        for key in ['label', 'prediction']:
            result[key] = kwargs['label_encoder'].inverse_transform(result[key])
        
        # # save image with true and predicted labels as name
        for i in range(len(result['image'])):
            img_arr = result['image'][i].transpose(1, 2, 0).astype(np.uint8)
            img = Image.fromarray(img_arr)
            img.save(os.path.join(path, f"{i}_{result['label'][i]}_{result['prediction'][i]}.jpg"))
        print("=> Done saving predictions")
