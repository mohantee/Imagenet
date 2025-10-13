import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from model import ResNet50

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations for validation
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load ImageNet validation dataset
    data_path = "path/to/imagenet"  # Update this path
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist. Please update the path.")

    val_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        transform=transform_test
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    model = ResNet50().to(device)
    
    # Load checkpoint
    checkpoint_path = "path/to/checkpoint.pth"  # Update this path
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print("No checkpoint found. Using untrained model.")

    # Evaluate
    accuracy = evaluate(model, val_loader, device)

if __name__ == '__main__':
    main()