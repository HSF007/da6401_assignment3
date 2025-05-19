import wandb
import os
import argparse
from config import Config
from train_attention import train_attention

sweep_attention_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'embed_size': {
            'values': [32, 64, 128, 256]
        },
        'hidden_size': {
            'values': [64, 128, 256, 512]
        },
        'num_encoder_layers': {
            'values': [1, 2]
        },
        'num_decoder_layers': {
            'values': [1, 2]
        },
        'cell_type': {
            'values': ['GRU', 'LSTM']
        },
        'dropout': {
            'values': [0.2, 0.3, 0.4]
        },
        'learning_rate': {
            'values': [0.0005, 0.001, 0.002]
        },
        'batch_size': {
            'values': [64, 128]
        },
        'teacher_forcing_ratio': {
            'values': [0.5, 0.7, 0.9]
        }
    }
}

def sweep_train_attention(config_defaults=None):
    # The issue is here - we need to ensure wandb.init() is called before using wandb.summary
    run = wandb.init(config=config_defaults)
    
    config = Config()
    
    # Update config with parameters from sweep
    wandb_config = wandb.config
    for key, value in wandb_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create model name based on parameters
    run_name = f"attn_{config.language}_{config.cell_type}_e{config.embed_size}_h{config.hidden_size}_en{config.num_encoder_layers}_de{config.num_decoder_layers}_d{config.dropout}_lr{config.learning_rate}"
            
    # Train the model
    # Don't initialize wandb again in train_attention
    model, accuracy = train_attention(config, run_name=run_name, init_wandb=False)
    
    # Log the final result
    wandb.run.summary['best_accuracy'] = accuracy
    # No need to call wandb.finish() as the context manager will handle it

def main():
    parser = argparse.ArgumentParser(description='Run W&B sweep for attention transliteration model')
    parser.add_argument('--sweep_id', type=str, help='The sweep ID to use (if already created)')
    parser.add_argument('--count', type=int, default=20, help='Number of sweep runs')
    parser.add_argument('--language', type=str, default='hi', 
                        choices=['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'si', 'ta', 'te', 'ur'],
                        help='Language code to use (default: hi)')
    args = parser.parse_args()
    
    # Set language in environment variable
    os.environ["TRANSLITERATION_LANGUAGE"] = args.language
    
    # Set up W&B project
    config = Config()
    
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep=sweep_attention_configuration, project=config.wandb_project, entity=config.wandb_entity)
    
    # Start sweep agent
    wandb.agent(sweep_id, function=sweep_train_attention, count=args.count)

if __name__ == "__main__":
    main()