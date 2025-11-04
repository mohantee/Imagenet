import torch
from tqdm import tqdm
import logging

def evaluate(model, test_loader, criterion, device, use_mixed_precision=False, dtype=torch.float16):
    """
    Evaluate model on validation/test set.
    
    NOTE: Always uses FP32 for evaluation to avoid numerical instability.
    Mixed precision is only used during training, not evaluation.
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    nan_count = 0
    inf_count = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # CRITICAL FIX: Always use FP32 for evaluation to prevent numerical instability
            # Mixed precision (BF16/FP16) can cause exploding loss values during evaluation
            with torch.amp.autocast('cuda', enabled=False, dtype=torch.float32):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Safety checks for NaN/Inf values
            if torch.isnan(loss) or torch.isinf(loss):
                if torch.isnan(loss):
                    nan_count += 1
                    logging.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
                if torch.isinf(loss):
                    inf_count += 1
                    logging.warning(f"Inf loss detected at batch {batch_idx}, skipping...")
                continue
            
            # Validate loss is reasonable (not exploding)
            loss_value = loss.item()
            if loss_value > 100.0:  # Unusually high loss
                logging.warning(f"Very high loss detected at batch {batch_idx}: {loss_value:.2f}")
            
            # Statistics
            test_loss += loss_value
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            avg_loss = test_loss / total if total > 0 else 0.0
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    # Report any issues
    if nan_count > 0:
        logging.warning(f"Evaluation encountered {nan_count} NaN loss values")
    if inf_count > 0:
        logging.warning(f"Evaluation encountered {inf_count} Inf loss values")
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    avg_loss = test_loss / total if total > 0 else 0.0
    logging.info(f'Test set: Average loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%')
    return accuracy

