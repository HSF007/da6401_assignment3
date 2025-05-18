import argparse
import os
import sys

from config import Config
from train import train
from evaluate import main as evaluate_main
from sweep import main as sweep_main

def check_dataset_exists(config):
    """Check if the Dakshina dataset exists in the expected location"""
    # Check if data files exist
    files_exist = (
        os.path.exists(config.train_file) and 
        os.path.exists(config.val_file) and 
        os.path.exists(config.test_file)
    )
    
    if not files_exist:
        print("\n" + "="*80)
        print(f"ERROR: Dakshina dataset files not found for language '{config.language}'!")
        print("="*80)
        print("\nThis program requires the Dakshina dataset to be downloaded and extracted.")
        print("\nPlease download the dataset from:")
        print("https://github.com/google-research-datasets/dakshina")
        print("\nExtract the dataset and make sure the following file structure exists:")
        print(f"  {config.data_dir}")
        print(f"  ├── {config.language}.translit.sampled.train.tsv")
        print(f"  ├── {config.language}.translit.sampled.dev.tsv")
        print(f"  └── {config.language}.translit.sampled.test.tsv")
        print("\nAlternatively, you can modify the paths in config.py to point to the correct location.")
        print("="*80 + "\n")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Transliteration Model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'sweep'],
                        help='Mode to run the script in: train, evaluate, or sweep')
    parser.add_argument('--language', type=str, default='hi', 
                        choices=['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'si', 'ta', 'te', 'ur'],
                        help='Language code to use (default: hi)')
    parser.add_argument('--sweep_id', type=str, help='The sweep ID to use (for sweep mode)')
    parser.add_argument('--count', type=int, default=50, help='Number of sweep runs (for sweep mode)')
    parser.add_argument('--use_sample_data', action='store_true', help='Create and use sample data for testing')
    
    # Model configuration arguments
    parser.add_argument('--embed_size', type=int, help='Embedding size')
    parser.add_argument('--hidden_size', type=int, help='Hidden layer size')
    parser.add_argument('--num_encoder_layers', type=int, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, help='Number of decoder layers')
    parser.add_argument('--cell_type', type=str, choices=['RNN', 'GRU', 'LSTM'], help='RNN cell type')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--teacher_forcing_ratio', type=float, help='Teacher forcing ratio')
    
    # Add new arguments for W&B configuration
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled'], 
                      default='online', help='W&B logging mode')
    
    args = parser.parse_args()
    
    # Set language in environment variable so config.py can access it
    os.environ["TRANSLITERATION_LANGUAGE"] = args.language
    
    # Load default configuration
    config = Config()
    
    # Update language in Config (in case it wasn't picked up by the environment)
    config.language = args.language
    
    # Check if dataset exists or use sample data
    if not check_dataset_exists(config):
        print("Exiting. Please download the dataset and try again.")
        sys.exit(1)
    
    # Update configuration with command-line arguments
    for arg in vars(args):
        if arg not in ['mode', 'sweep_id', 'count', 'use_sample_data', 'wandb_mode', 'language'] and getattr(args, arg) is not None:
            setattr(config, arg, getattr(args, arg))
    
    # Set W&B mode
    os.environ["WANDB_MODE"] = args.wandb_mode
    
    # Run in selected mode
    if args.mode == 'train':
        train(config)
    elif args.mode == 'evaluate':
        sys.argv = [sys.argv[0]]
        if args.sweep_id:
            sys.argv.extend(['--sweep_id', args.sweep_id])
        # Pass language to evaluate.py through environment variable
        evaluate_main()
    elif args.mode == 'sweep':
        # Pass arguments to sweep_main
        sys.argv = [sys.argv[0]]
        if args.sweep_id:
            sys.argv.extend(['--sweep_id', args.sweep_id])
        if args.count:
            sys.argv.extend(['--count', str(args.count)])
        if args.language:
            sys.argv.extend(['--language', args.language])
        sweep_main()

if __name__ == "__main__":
    main()