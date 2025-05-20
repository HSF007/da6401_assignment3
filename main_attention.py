import argparse
import os
import sys

from config import Config
from train_attention import train_attention
from evaluation_attention import main as evaluate_attention_main
from sweep_attention import main as sweep_attention_main

def check_dataset_exists(config):
    """Check if the Dakshina dataset exists in the expected location"""
    files_exist = (
        os.path.exists(config.train_file) and 
        os.path.exists(config.val_file) and 
        os.path.exists(config.test_file)
    )
    
    if not files_exist:
        print(f"\nERROR: Dakshina dataset files not found for language '{config.language}'!")
        print("\nPlease download the dataset from https://github.com/google-research-datasets/dakshina")
        print(f"\nMake sure the following files exist:")
        print(f"  {config.train_file}")
        print(f"  {config.val_file}")
        print(f"  {config.test_file}\n")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Attention-based Transliteration Model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'sweep'],
                        help='Mode to run the script in: train, evaluate, or sweep')
    parser.add_argument('--language', type=str, default='hi', 
                        choices=['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'si', 'ta', 'te', 'ur'],
                        help='Language code to use (default: hi)')
    parser.add_argument('--sweep_id', type=str, help='The sweep ID to use (for sweep mode)')
    parser.add_argument('--count', type=int, default=20, help='Number of sweep runs (for sweep mode)')
    
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
    
    # W&B configuration
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled'], 
                      default='online', help='W&B logging mode')
    
    args = parser.parse_args()
    
    # Set language in environment variable
    os.environ["TRANSLITERATION_LANGUAGE"] = args.language
    
    # Load default configuration
    config = Config()
    
    # Update language in Config
    config.language = args.language
    
    # Check if dataset exists
    if not check_dataset_exists(config):
        print("Exiting. Please download the dataset and try again.")
        sys.exit(1)
    
    # Update configuration with command-line arguments
    for arg in vars(args):
        if arg not in ['mode', 'sweep_id', 'count', 'wandb_mode', 'language'] and getattr(args, arg) is not None:
            setattr(config, arg, getattr(args, arg))
    
    # Set W&B mode
    os.environ["WANDB_MODE"] = args.wandb_mode
    
    # Run in selected mode
    if args.mode == 'train':
        train_attention(config)
    elif args.mode == 'evaluate':
        sys_args = [sys.argv[0]]
        if args.sweep_id:
            sys_args.extend(['--sweep_id', args.sweep_id])
        
        original_argv = sys.argv
        sys.argv = sys_args
        evaluate_attention_main()
        sys.argv = original_argv
    elif args.mode == 'sweep':
        # Pass arguments to sweep
        sys_args = [sys.argv[0]]
        if args.sweep_id:
            sys_args.extend(['--sweep_id', args.sweep_id])
        if args.count:
            sys_args.extend(['--count', str(args.count)])
        if args.language:
            sys_args.extend(['--language', args.language])
        
        # Replace argv temporarily
        original_argv = sys.argv
        sys.argv = sys_args
        sweep_attention_main()
        sys.argv = original_argv

if __name__ == "__main__":
    main()