import torch
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (windows, labels) in enumerate(tqdm(dataloader, desc="Training")):
        windows = windows.to(device)
        
        # Forward pass
        outputs = model(windows)
        
        # Compute loss
        losses = criterion(outputs, labels)
        loss = losses['loss_total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for windows, _ in tqdm(dataloader, desc="Evaluating"):
            windows = windows.to(device)
            outputs = model(windows)
            
            # Get predictions (queries with class != no-object)
            pred_logits = outputs['class_logits']
            pred_classes = pred_logits.argmax(-1)
            pred_spans = outputs['spans']
            
            for b in range(windows.size(0)):
                valid_queries = pred_classes[b] != model.num_classes
                
                if valid_queries.any():
                    predictions.append({
                        'classes': pred_classes[b][valid_queries].cpu().numpy(),
                        'spans': pred_spans[b][valid_queries].cpu().numpy(),
                        'scores': pred_logits[b][valid_queries].softmax(-1).max(-1)[0].cpu().numpy()
                    })
    
    return predictions