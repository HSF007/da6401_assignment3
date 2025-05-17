import torch
import os
import numpy as np

def calculate_accuracy(predictions, targets, target_lengths, ignore_index=0):
    """
    Calculate the accuracy of the predictions against the targets.
    A prediction is considered correct only if all characters in the sequence match.
    """
    batch_size = predictions.size(0)
    correct = 0
    
    for i in range(batch_size):
        pred_seq = predictions[i].cpu().numpy()
        target_seq = targets[i, 1:target_lengths[i]-1].cpu().numpy()  # Exclude SOS and EOS tokens
        
        # Check if prediction exactly matches target
        if len(pred_seq) == len(target_seq) and np.array_equal(pred_seq, target_seq):
            correct += 1
    
    return correct / batch_size

def save_checkpoint(model, optimizer, epoch, accuracy, filename):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }, filename)

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint.get('accuracy', 0.0)
    return model, optimizer, epoch, accuracy

def indices_to_string(indices, idx_to_char, eos_idx=None):
    """Convert a sequence of indices to a string"""
    if eos_idx is not None:
        # Find the index of the first EOS token
        try:
            eos_pos = indices.index(eos_idx)
            indices = indices[:eos_pos]
        except ValueError:
            pass  # No EOS token found
    
    return ''.join([idx_to_char[idx] for idx in indices if idx in idx_to_char and idx_to_char[idx] not in ['<PAD>', '< SOS >', '<EOS>']])

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)