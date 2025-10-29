import torch
from tqdm import tqdm
import logging

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
            with torch.amp.autocast('cuda', enabled=use_mixed_precision, dtype=dtype):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Statistics: accumulate loss weighted by batch size so final avg is per-sample
            batch_size = inputs.size(0)
            test_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

            # Update progress bar using per-sample average loss
            avg_loss = test_loss / total if total > 0 else 0.0
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    accuracy = 100. * correct / total
    logging.info(f'Test set: Average loss: {test_loss/total:.3f}, Accuracy: {accuracy:.2f}%')
    return accuracy

