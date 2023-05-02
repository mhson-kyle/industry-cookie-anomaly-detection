from dataset import get_dataset
from tqdm.auto import tqdm
import os
import torch
import argparse
from model import ResNetDefectClassifier, CNNDefectClassifier
from torch import nn, optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
    
def train(model, train_dataset, val_dataset, batch_size=32, num_epochs=30, learning_rate = 0.001, patience=5, device=device, save=True, save_name='model'):
    train_dataloader = torch.utils.data.DataLoader(train_dataset,         
                                           batch_size=batch_size,
                                           shuffle=True)    

    val_dataloader = torch.utils.data.DataLoader(val_dataset,         
                                           batch_size=batch_size,
                                           shuffle=True)   

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(lr=learning_rate, params=model.parameters())
    early_stopping = EarlyStopping(patience=3)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training", unit="batch")
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        progress_bar = tqdm(val_dataloader, desc="Validating", unit="batch")
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
    
        val_loss = val_loss / len(val_dataloader.dataset)
        val_acc = val_corrects.double() / len(val_dataloader.dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Validation Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if early_stopping(val_loss):
            print("Early stopping triggered. Stopping training.")
            break
    if save == True:
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), f'model/{save_name}-defect-classifier.pth')
    return model, (train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="resnet", help="Model to use for training")
    argparser.add_argument('--trainable', type=bool, default=False, help='Whether to train the entire model or just the last layer')
    argparser.add_argument('--train_path', type=str, default='data/images/train/', help='Path to training data')
    argparser.add_argument('--val_path', type=str, default='data/images/valid/', help='Path to validation data')
    argparser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    argparser.add_argument("--learning_rate", type=float, default=0.001, help="Learning Rate for training")
    argparser.add_argument("--patience", type=int, default=5, help="Patience for training")
    argparser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to training")
    argparser.add_argument("--save_name", type=str, default='model', help="Save name")
    args = argparser.parse_args()

    train_dataset = get_dataset(args.train_path)
    valid_dataset = get_dataset(args.val_path)
    
    if args.model == "resnet":
        model = ResNetDefectClassifier(trainable=args.trainable)
    elif args.model == "cnn":
        model = CNNDefectClassifier()
        
    train(model, train_dataset, valid_dataset, batch_size=args.batch_size, learning_rate=args.learning_rate, patience=args.patience, num_epochs=args.num_epochs, device=device, save=True, save_name=args.save_name)
