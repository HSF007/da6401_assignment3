import wandb
import os
import argparse
from config import Config
from train import train

sweep_configuration = {
    'method': 'bayes',  # or 'grid' or 'random'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'embed_size': {
            'values': [16, 32, 64, 128, 256]
        },
        'hidden_size': {
            'values': [32, 64, 128, 256, 512]
        },
        'num_encoder_layers': {
            'values': [1, 2, 3]
        },
        'num_decoder_layers': {
            'values': [1, 2, 3]
        },
        'cell_type': {
            'values': ['RNN', 'GRU', 'LSTM']
        },
        'dropout': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'learning_rate': {
            'values': [0.0001, 0.0005, 0.001, 0.005]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'teacher_forcing_ratio': {
            'values': [0.3, 0.5, 0.7, 0.9]
        }
    }
}

def sweep_train(config_defaults=None):
    # Initialize a new wandb run
    with wandb.init(config=config_defaults) as run:
        # Copy hyperparameters from wandb config
        config = Config()
        
        # Update config with parameters from sweep
        wandb_config = wandb.config
        for key, value in wandb_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Create model name based on parameters
        run_name = f"{config.language}_{config.cell_type}_e{config.embed_size}_h{config.hidden_size}_en{config.num_encoder_layers}_de{config.num_decoder_layers}_d{config.dropout}_lr{config.learning_rate}"
                
        # Train the model
        # Note: We don't pass run_name here as wandb.init() is already handled in the train function
        # and we're using the current run within this context
        model, accuracy = train(config, run_already_initialized=True)
        
        # Log the final result to the current run
        run.summary['best_accuracy'] = accuracy

def main():
    parser = argparse.ArgumentParser(description='Run W&B sweep for transliteration model')
    parser.add_argument('--sweep_id', type=str, help='The sweep ID to use (if already created)')
    parser.add_argument('--count', type=int, default=50, help='Number of sweep runs')
    parser.add_argument('--language', type=str, default='hi', 
                        choices=['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'si', 'ta', 'te', 'ur'],
                        help='Language code to use (default: hi)')
    args = parser.parse_args()
    
    # Set language in environment variable so config.py can access it
    os.environ["TRANSLITERATION_LANGUAGE"] = args.language
    
    # Set up W&B project
    config = Config()
    
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=config.wandb_project, entity=config.wandb_entity)
    
    # Start sweep agent
    wandb.agent(sweep_id, function=sweep_train, count=args.count)

if __name__ == "__main__":
    main()