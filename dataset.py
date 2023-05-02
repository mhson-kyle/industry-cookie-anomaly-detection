import torchvision.transforms as transforms
from torchvision import datasets

def get_dataset(data_path):
    dataset = datasets.ImageFolder(root=data_path, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])) 
    return dataset
