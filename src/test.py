import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from model import ResNet50

def evaluate(model, test_loader, criterion, device, use_mixed_precision=False, dtype=torch.float16):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    # Check CUDA availability for mixed precision
    if use_mixed_precision and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Mixed precision evaluation will be disabled.")
        use_mixed_precision = False

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixed precision context for evaluation
            with torch.cuda.amp.autocast(enabled=use_mixed_precision, dtype=dtype):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{test_loss/total:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss/total:.3f}, '
          f'Accuracy: {accuracy:.2f}%')
    return accuracy

