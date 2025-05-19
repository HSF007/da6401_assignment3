import torch
import wandb
import os
import time
from tqdm import tqdm
import random

from config import Config
from dataloader import get_dataloader
from seq2seq_attention import Seq2SeqAttention
from utils import save_checkpoint, create_directory, indices_to_string

def train_attention(config, run_name=None, init_wandb=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Only initialize wandb if specified
    if init_wandb:
        if run_name:
            wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=run_name, config=vars(config))
        else:
            default_run_name = f"{config.language}_attn_{config.cell_type}_e{config.embed_size}_h{config.hidden_size}"
            wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=default_run_name, config=vars(config))
        
        wandb.config.update({"language": config.language})
    
    print(f"\nTraining attention-based transliteration model for language: {config.language}")
    print(f"Using dataset from: {config.data_dir}\n")
    
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
    
    model = Seq2SeqAttention(
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
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    create_directory(config.model_dir)
    
    best_accuracy = 0.0
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            source_lengths = batch['source_lengths']
            
            optimizer.zero_grad()
            
            output, _ = model(source, target, source_lengths, config.teacher_forcing_ratio)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target = target[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss, val_accuracy = evaluate_attention(model, val_loader, criterion, device, train_dataset)
        
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'language': config.language
        })
        
        print(f'Epoch: {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if not os.path.exists(os.path.join(config.model_dir)):
                os.makedirs(os.path.join(config.model_dir))

            model_path = os.path.join(config.model_dir, f'best_attention_model_{config.language}_{wandb.run.id}.pt')
            save_checkpoint(
                model, 
                optimizer, 
                epoch, 
                val_accuracy, 
                model_path
            )
            
            try:
                wandb_model_path = os.path.join(wandb.run.dir, "best_attention_model.pt")
                torch.save(model.state_dict(), wandb_model_path)
                
                artifact = wandb.Artifact(f"attention-model-{config.language}-{wandb.run.id}", type="model")
                artifact.add_file(wandb_model_path)
                wandb.log_artifact(artifact)
                
                print(f"Model successfully saved to W&B as artifact")
            except Exception as e:
                print(f"Failed to save model to W&B: {str(e)}")
                print(f"Model was saved locally to {model_path}")
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.model_dir, f'checkpoint_attention_{config.language}_{epoch+1}_{wandb.run.id}.pt')
            save_checkpoint(
                model, 
                optimizer, 
                epoch, 
                val_accuracy, 
                checkpoint_path
            )
    
    # Only finish wandb if we initialized it
    if init_wandb:
        wandb.finish()
        
    return model, best_accuracy

def evaluate_attention(model, dataloader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            source_lengths = batch['source_lengths']
            
            output, _ = model(source, target, source_lengths, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            output_for_loss = output[:, 1:].reshape(-1, output_dim)
            target_for_loss = target[:, 1:].reshape(-1)
            
            loss = criterion(output_for_loss, target_for_loss)
            total_loss += loss.item()
            
            predictions, _ = model.predict(
                source,
                source_lengths,
                dataset.target_vocab_size,
                dataset.target_char_to_idx['< SOS >'],
                dataset.target_char_to_idx['<EOS>']
            )
            
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
    train_attention(config)