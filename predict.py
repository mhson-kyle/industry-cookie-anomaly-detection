import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
from model import ResNetDefectClassifier, CNNDefectClassifier
import argparse
from dataset import get_dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(model, test_dataset, device=device):
    test_dataloader = torch.utils.data.DataLoader(test_dataset,         
                                           batch_size=1,
                                           shuffle=True)  
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    y_pred = []
    y_true = []
    progress_bar = tqdm(test_dataloader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.append(int(preds.data.cpu()[0]))
            y_true.append(int(labels.data.cpu()[0]))
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            
    print(classification_report(y_true, y_pred))
    test_loss = test_loss / len(test_dataloader.dataset)
    test_acc = test_corrects.double() / len(test_dataloader.dataset)
    print(f"Evaluating Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    return (test_acc, test_loss)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="resnet", help="Model to use for predicting")
    argparser.add_argument('--model_path', type=str, default='model/resnet-defect-classifier.pth', help='Path to trained model weight')
    argparser.add_argument('--test_path', type=str, default='data/images/test/', help='Path to predicting data')
    args = argparser.parse_args()

    if args.model == 'resnet':
        model = ResNetDefectClassifier()
    elif args.model == 'cnn':
        model = CNNDefectClassifier()
        
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    test_dataset = get_dataset(args.test_path)
    predict(model, test_dataset=test_dataset)
    