import torch
from tqdm import tqdm
import logging

import model

def evaluate(model, test_loader, criterion, device, use_mixed_precision=False, dtype=torch.float16):
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=use_mixed_precision, dtype=dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            batch_size = inputs.size(0)
            test_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({
                'loss': f'{test_loss / total:.3f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    accuracy = 100. * correct / total
    logging.info(f'Test set: Average loss: {test_loss / total:.3f}, Accuracy: {accuracy:.2f}%')
    return accuracy

