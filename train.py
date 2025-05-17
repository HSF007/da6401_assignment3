import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import time
from tqdm import tqdm

from config import Config
from dataloader import get_dataloader
from seq2seq import Seq2Seq
from utils import save_checkpoint, create_directory, indices_to_string

def train(config, run_name=None, run_already_initialized=False):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize W&B only if it's not already initialized
    if not run_already_initialized:
        if run_name:
            wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=run_name, config=vars(config))
        else:
            # Include language in the run name if not provided
            default_run_name = f"{config.language}_{config.cell_type}_e{config.embed_size}_h{config.hidden_size}"
            wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=default_run_name, config=vars(config))
    
        # Explicitly log the language being used
        wandb.config.update({"language": config.language})
    
    print(f"\nTraining transliteration model for language: {config.language}")
    print(f"Using dataset from: {config.data_dir}\n")
    
    try:
        # Load data
        train_loader, train_dataset = get_dataloader(
            config.train_file, 
            config.batch_size, 
            shuffle=True
        )
        
        val_loader, val_dataset = get_dataloader(
            config.val_file, 
            config.batch_size, 
            shuffle=False
        )
        
        # Create model
        model = Seq2Seq(
            input_size=train_dataset.source_vocab_size,
            output_size=train_dataset.target_vocab_size,
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dropout=config.dropout,
            cell_type=config.cell_type
        )
        
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Create directory for saving models
        create_directory(config.model_dir)
        
        best_accuracy = 0.0
        
        # Training loop
        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
            
            for batch_idx, batch in enumerate(pbar):
                # Move tensors to device
                source = batch['source'].to(device)
                target = batch['target'].to(device)
                source_lengths = batch['source_lengths']
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = model(source, target, source_lengths, config.teacher_forcing_ratio)
                
                # Calculate loss
                # Reshape output and target for loss calculation
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)  # Remove SOS token
                target = target[:, 1:].reshape(-1)  # Remove SOS token
                
                # Calculate loss
                loss = criterion(output, target)
                
                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            # Calculate average loss
            avg_loss = total_loss / len(train_loader)
            
            # Evaluate on validation set
            model.eval()
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, train_dataset)
            
            # Log metrics to W&B
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'language': config.language  # Log language with metrics
            })
            
            print(f'Epoch: {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            
            # Save model if it's the best so far
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                if not os.path.exists(os.path.join(config.model_dir)):
                    os.makedirs(os.path.join(config.model_dir))

                # Save locally first with language code in filename
                model_path = os.path.join(config.model_dir, f'best_model_{config.language}_{wandb.run.id}.pt')
                save_checkpoint(
                    model, 
                    optimizer, 
                    epoch, 
                    val_accuracy, 
                    model_path
                )
                
                # Save model state dictionary to W&B
                try:
                    # Save model for W&B
                    wandb_model_path = os.path.join(wandb.run.dir, "best_model.pt")
                    torch.save(model.state_dict(), wandb_model_path)
                    
                    # Use W&B Artifact API instead of direct file save
                    artifact = wandb.Artifact(f"model-{config.language}-{wandb.run.id}", type="model")
                    artifact.add_file(wandb_model_path)
                    wandb.log_artifact(artifact)
                    
                    print(f"Model successfully saved to W&B as artifact")
                except Exception as e:
                    print(f"Failed to save model to W&B: {str(e)}")
                    print(f"Model was saved locally to {model_path}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(config.model_dir, f'checkpoint_{config.language}_{epoch+1}_{wandb.run.id}.pt')
                save_checkpoint(
                    model, 
                    optimizer, 
                    epoch, 
                    val_accuracy, 
                    checkpoint_path
                )

        # Only finish wandb if we started it in this function
        if not run_already_initialized:
            wandb.finish()
        
        return model, best_accuracy
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Only finish wandb if we started it in this function
        if not run_already_initialized:
            wandb.finish()
        # Re-raise the exception for proper error handling
        raise

def evaluate(model, dataloader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move tensors to device
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            source_lengths = batch['source_lengths']
            target_lengths = batch['target_lengths']
            
            # Forward pass
            output = model(source, target, source_lengths, teacher_forcing_ratio=0)
            
            # Calculate loss
            output_dim = output.shape[-1]
            output_for_loss = output[:, 1:].reshape(-1, output_dim)  # Remove SOS token
            target_for_loss = target[:, 1:].reshape(-1)  # Remove SOS token
            
            loss = criterion(output_for_loss, target_for_loss)
            total_loss += loss.item()
            
            # Generate predictions
            predictions = model.predict(
                source,
                source_lengths,
                dataset.target_vocab_size,
                dataset.target_char_to_idx['< SOS >'],
                dataset.target_char_to_idx['<EOS>']
            )
            
            # Calculate accuracy (exact match)
            for i in range(len(predictions)):
                pred_seq = [idx.item() for idx in predictions[i]]
                pred_str = indices_to_string(pred_seq, dataset.target_idx_to_char, dataset.target_char_to_idx['<EOS>'])
                target_str = batch['target_texts'][i]
                
                if pred_str == target_str:
                    total_correct += 1
            
            total_examples += len(predictions)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_examples
    
    return avg_loss, accuracy

if __name__ == "__main__":
    config = Config()
    train(config)