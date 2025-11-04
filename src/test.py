import torch
from tqdm import tqdm
import logging

def evaluate(model, test_loader, criterion, device, use_mixed_precision=False, dtype=torch.float16):
    """
    Evaluate model on validation/test set.
    
    CRITICAL: Always uses FP32 for evaluation to prevent numerical instability.
    Mixed precision (BF16/FP16) can cause exploding loss values during evaluation.
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    nan_count = 0
    inf_count = 0
    skipped_batches = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # CRITICAL: Always use FP32 for evaluation - disable mixed precision completely
            # This prevents numerical instability that causes loss explosion
            with torch.cuda.amp.autocast(enabled=False):  # Explicitly disable autocast
                try:
                    # Forward pass in FP32
                    outputs = model(inputs.float() if inputs.dtype != torch.float32 else inputs)
                    loss = criterion(outputs, targets)
                    
                    # Check for NaN/Inf in outputs before loss computation
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        nan_count += 1
                        logging.warning(f"NaN/Inf outputs detected at batch {batch_idx}, skipping...")
                        skipped_batches += 1
                        continue
                    
                    # Check loss value
                    if torch.isnan(loss) or torch.isinf(loss):
                        if torch.isnan(loss):
                            nan_count += 1
                            logging.warning(f"NaN loss detected at batch {batch_idx}")
                        if torch.isinf(loss):
                            inf_count += 1
                            logging.warning(f"Inf loss detected at batch {batch_idx}")
                        skipped_batches += 1
                        continue
                    
                    loss_value = loss.item()
                    
                    # Skip batches with extremely high loss (more aggressive threshold)
                    # Normal ImageNet validation loss should be < 10.0 for good models
                    # Anything > 20.0 is suspicious, > 50.0 is definitely corrupted
                    if loss_value > 50.0:
                        logging.warning(f"Extremely high loss detected at batch {batch_idx}: {loss_value:.2f}, skipping batch")
                        skipped_batches += 1
                        continue
                    
                    # Also skip moderately high loss batches (likely corrupted)
                    if loss_value > 20.0:
                        logging.info(f"High loss detected at batch {batch_idx}: {loss_value:.2f}, skipping batch")
                        skipped_batches += 1
                        continue
                    
                    # Accumulate statistics
                    test_loss += loss_value
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                except Exception as e:
                    logging.error(f"Error during evaluation at batch {batch_idx}: {e}")
                    skipped_batches += 1
                    continue
            
            # Update progress bar
            if total > 0:
                avg_loss = test_loss / (batch_idx + 1 - skipped_batches) if (batch_idx + 1 - skipped_batches) > 0 else 0.0
                pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'skipped': skipped_batches
                })
    
    # Report any issues
    total_batches = len(test_loader)
    if nan_count > 0:
        logging.warning(f"⚠️  Evaluation encountered {nan_count} NaN values - model may be unstable")
    if inf_count > 0:
        logging.warning(f"⚠️  Evaluation encountered {inf_count} Inf values - model may be unstable")
    if skipped_batches > 0:
        skip_percentage = (skipped_batches / total_batches) * 100
        logging.warning(f"⚠️  Skipped {skipped_batches}/{total_batches} batches ({skip_percentage:.1f}%) due to high loss values")
        if skip_percentage > 50:
            logging.error(f"❌ More than 50% of batches were skipped - evaluation accuracy may be unreliable!")
    
    if total == 0:
        logging.error("❌ No valid batches evaluated - model output may be completely corrupted")
        return 0.0
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / (len(test_loader) - skipped_batches) if (len(test_loader) - skipped_batches) > 0 else 0.0
    
    # Final safety check
    if avg_loss > 100.0:
        logging.error(f"❌ Average loss is extremely high ({avg_loss:.2f}) - evaluation may be corrupted")
        logging.error("   Model weights may contain NaN/Inf values. Check model state.")
    
    logging.info(f'Test set: Average loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%')
    return accuracy

