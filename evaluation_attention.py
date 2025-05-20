import torch
import wandb
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
import argparse
import sys

from config import Config
from dataloader import get_dataloader
from seq2seq_attention import Seq2SeqAttention
from utils import indices_to_string, create_directory

def setup_fonts_for_indic_scripts():
    """Setup fonts for Indic scripts like Devanagari"""
    # Try to find a suitable font that supports Devanagari
    font_paths = []
    
    # Common Devanagari font names on different systems
    devanagari_fonts = [
        # Linux fonts
        '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
        '/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf',
        # Windows fonts
        'C:/Windows/Fonts/mangal.ttf',
        'C:/Windows/Fonts/aparaj.ttf',
        # macOS fonts
        '/Library/Fonts/Devanagari.ttf',
        '/System/Library/Fonts/Supplemental/Devanagari MT.ttf',
    ]
    
    # Check if any of these fonts exist
    for font_path in devanagari_fonts:
        if os.path.exists(font_path):
            font_paths.append(font_path)
    
    if font_paths:
        # Use the first found font
        print(f"Using font: {font_paths[0]} for Devanagari characters")
        devanagari_font = fm.FontProperties(fname=font_paths[0])
        plt.rcParams['font.family'] = 'sans-serif'
        return devanagari_font
    else:
        print("Warning: No suitable Devanagari font found. Characters may not display correctly.")
        return None

def evaluate_model_with_attention(model, dataloader, dataset, device, config):
    model.eval()
    results = []
    total_correct = 0
    total_examples = 0
    all_attention_maps = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            source = batch['source'].to(device)
            source_lengths = batch['source_lengths']
            source_texts = batch['source_texts']
            target_texts = batch['target_texts']
            
            predictions, attention_weights = model.predict(
                source,
                source_lengths,
                dataset.target_vocab_size,
                dataset.target_char_to_idx['< SOS >'],
                dataset.target_char_to_idx['<EOS>']
            )
            
            for i in range(len(predictions)):
                pred_seq = [idx.item() for idx in predictions[i]]
                pred_str = indices_to_string(pred_seq, dataset.target_idx_to_char, 
                                            dataset.target_char_to_idx['<EOS>'])
                
                is_correct = pred_str == target_texts[i]
                if is_correct:
                    total_correct += 1
                
                # Convert source text to list of characters for attention visualization
                source_chars = [c for c in source_texts[i]]
                
                # Get output characters
                output_chars = [c for c in pred_str]
                
                # Get attention map for this example
                attn_map = attention_weights[i, :len(output_chars), :len(source_chars)].cpu().numpy()
                
                results.append({
                    'source': source_texts[i],
                    'target': target_texts[i],
                    'prediction': pred_str,
                    'correct': is_correct,
                    'source_chars': source_chars,
                    'output_chars': output_chars,
                    'attention_map': attn_map.tolist()
                })
                
                all_attention_maps.append({
                    'source': source_chars,
                    'output': output_chars,
                    'map': attn_map
                })
                
            total_examples = len(predictions)
    
    accuracy = total_correct / total_examples
    
    return results, accuracy, all_attention_maps

def get_best_attention_model(config, device, sweep_id=None):
    """Get the best model configuration from a completed sweep"""

    if not sweep_id:
        raise ValueError("Sweep ID is required to get the best model configuration.")
    
    api = wandb.Api()

    try:
        sweep = api.sweep(f"{config.wandb_entity}/{config.wandb_project}/{sweep_id}")

        best_val_acc = -1
        best_run = None

        for run in sweep.runs:
            val_acc = run.summary.get("val_accuracy")
            if val_acc is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_run = run
        
        if best_run is None:
            raise ValueError("No runs with 'val_accuracy' found in the sweep")
        
        # Extract hyperparameters from the best run
        best_config = {
            'embed_size': best_run.config.get('embed_size', config.embed_size),
            'hidden_size': best_run.config.get('hidden_size', config.hidden_size),
            'num_encoder_layers': best_run.config.get('num_encoder_layers', config.num_encoder_layers),
            'num_decoder_layers': best_run.config.get('num_decoder_layers', config.num_decoder_layers),
            'cell_type': best_run.config.get('cell_type', config.cell_type),
            'dropout': best_run.config.get('dropout', config.dropout)
        }

        # Update config with best hyperparameters
        for key, value in best_config.items():
            setattr(config, key, value)
        
        print("Best model hyperparameters:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        
        # Try to download the best model
        try:
            # Try to find artifact by name
            artifact = api.artifact(f"{config.wandb_entity}/{config.wandb_project}/attention-model-{config.language}-{best_run.id}:latest")
            artifact_dir = artifact.download()
            best_model_path = os.path.join(artifact_dir, "best_attention_model.pt")
            
            # If artifact download failed, try to get the file directly
            if not os.path.exists(best_model_path):
                best_model_path = best_run.file("best_attention_model.pt").download()
            
            print(f"Downloaded best model from W&B")
        except Exception as e:
            print(f"Failed to download model from W&B: {str(e)}")
            raise FileNotFoundError("Could not download the best model from W&B.")
    except Exception as e:
        print(f"Failed to get best model from W&B: {str(e)}")
    
    return best_config, best_model_path

def plot_attention_heatmap(attention_map, source_tokens, output_tokens, ax=None, font_prop=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        
    im = ax.imshow(attention_map, cmap='viridis')
    
    # Set x and y labels with font property for Devanagari
    ax.set_xticks(np.arange(len(source_tokens)))
    ax.set_yticks(np.arange(len(output_tokens)))
    
    # Use ASCII transliteration if font is not available
    if font_prop is None:
        # Use transliteration or character codes if no proper font
        x_labels = [f"'{token}'" for token in source_tokens]
        y_labels = [f"'{token}'" for token in output_tokens]
    else:
        x_labels = source_tokens
        y_labels = output_tokens
    
    ax.set_xticklabels(x_labels, rotation=90, fontproperties=font_prop)
    ax.set_yticklabels(y_labels, fontproperties=font_prop)
    
    # Show grid lines
    ax.set_xticks(np.arange(-.5, len(source_tokens), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(output_tokens), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    # Color bar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations (with numeric values only)
    for i in range(len(output_tokens)):
        for j in range(len(source_tokens)):
            text = ax.text(j, i, f"{attention_map[i, j]:.2f}",
                          ha="center", va="center", color="w" if attention_map[i, j] > 0.5 else "black")
    
    ax.set_title("Attention Heatmap")
    
    return ax

def plot_connectivity_visualization(attention_map, source_tokens, output_tokens, ax=None, font_prop=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    y_source = np.arange(len(source_tokens))
    y_output = np.arange(len(output_tokens))
    
    # Plot source and output tokens
    ax.scatter([0] * len(source_tokens), y_source, marker='o', s=100, color='blue', label='Source')
    ax.scatter([1] * len(output_tokens), y_output, marker='o', s=100, color='red', label='Output')
    
    # Add token labels with font property for Devanagari
    for i, token in enumerate(source_tokens):
        ax.annotate(token, xy=(0, i), xytext=(-0.15, i), fontsize=10, fontproperties=font_prop)
    
    for i, token in enumerate(output_tokens):
        ax.annotate(token, xy=(1, i), xytext=(1.05, i), fontsize=10, fontproperties=font_prop)
    
    # Draw attention connections
    max_weights = np.argmax(attention_map, axis=1)
    for i in range(len(output_tokens)):
        max_idx = max_weights[i]
        weight = attention_map[i, max_idx]
        ax.plot([0, 1], [max_idx, i], 'k-', alpha=weight, linewidth=weight*3)
    
    # Set plot limits and labels
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.5, max(len(source_tokens), len(output_tokens)) - 0.5)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Token Index')
    ax.set_title('Attention Connectivity Visualization')
    ax.legend()
    
    # Remove axis ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Source', 'Output'])
    ax.set_yticks([])
    
    return ax

def main():
    parser = argparse.ArgumentParser(description='Run evaluation for best model in a sweep')
    parser.add_argument('--sweep_id', type=str, help='The sweep ID to use (if already created)')
    parser.add_argument('--font', type=str, help='Path to font file that supports Devanagari', default=None)
    parser.add_argument('--save_figs', action='store_true', help='Save figures locally in addition to uploading to W&B')

    args = parser.parse_args()
    
    # Setup font for Devanagari characters
    devanagari_font = None
    if args.font and os.path.exists(args.font):
        devanagari_font = fm.FontProperties(fname=args.font)
        print(f"Using specified font: {args.font}")
    else:
        devanagari_font = setup_fonts_for_indic_scripts()
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sweep_id = args.sweep_id
    
    # Create attention prediction directory
    attention_pred_dir = os.path.join(os.path.dirname(config.prediction_dir), "predictions_attention")
    create_directory(attention_pred_dir)
    
    # Always save figures - no need to make it conditional since that's what the user wants
    # Create directory for saving figures in the same directory as predictions
    fig_dir = os.path.join(attention_pred_dir, "figures")
    create_directory(fig_dir)
    
    # Temporarily override the prediction directory
    original_pred_dir = config.prediction_dir
    config.prediction_dir = attention_pred_dir

    # Initialize W&B
    wandb.init(project=config.wandb_project, entity=config.wandb_entity, job_type="attention_eval", 
               name=f"attn_eval_{config.language}")
    
    # Log the language being evaluated
    wandb.config.update({"language": config.language})
    
    # Load test data
    test_loader, test_dataset = get_dataloader(
        config.test_file, 
        config.batch_size, 
        shuffle=False
    )
    
    # Get the best model from sweep
    best_config, best_model_path = get_best_attention_model(config, device, sweep_id)

    # Load the model checkpoint to get the correct vocabulary size
    checkpoint = torch.load(best_model_path, map_location=device)

    # Extract the vocabulary size from the model checkpoint
    output_size = checkpoint['decoder.fc_out.bias'].size(0)

    # Create model with attention
    model = Seq2SeqAttention(
        input_size=test_dataset.source_vocab_size,
        output_size=output_size,
        embed_size=best_config['embed_size'],
        hidden_size=best_config['hidden_size'],
        num_encoder_layers=best_config['num_encoder_layers'],
        num_decoder_layers=best_config['num_decoder_layers'],
        dropout=best_config['dropout'],
        cell_type=best_config['cell_type']
    )
    
    # Load the model weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate model
    results, accuracy, attention_maps = evaluate_model_with_attention(model, test_loader, test_dataset, device, config)
    
    # Save results
    run_id = wandb.run.id
    output_file = os.path.join(config.prediction_dir, f"predictions_attention_{config.language}_{run_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Log accuracy to W&B
    wandb.log({'test_accuracy': accuracy, 'language': config.language})
    
    print(f'Test Accuracy for {config.language}: {accuracy:.4f}')
    print(f'Predictions saved to {output_file}')
    
    # Plot attention heatmaps for sample predictions using Devanagari font
    num_samples = min(9, len(attention_maps))
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_samples):
        sample = attention_maps[i]
        plot_attention_heatmap(
            sample['map'], 
            sample['source'], 
            sample['output'], 
            ax=axes[i],
            font_prop=devanagari_font
        )
    
    plt.tight_layout()
    # Log to W&B
    wandb.log({"attention_heatmaps": wandb.Image(fig)})
    
    # Always save figures locally
    heatmap_fig_path = os.path.join(fig_dir, f"attention_heatmaps_{config.language}_{run_id}.png")
    fig.savefig(heatmap_fig_path, dpi=300, bbox_inches='tight')
    print(f"Attention heatmaps saved to {heatmap_fig_path}")
    
    # Plot connectivity visualizations for sample predictions
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_samples):
        sample = attention_maps[i]
        plot_connectivity_visualization(
            sample['map'], 
            sample['source'], 
            sample['output'], 
            ax=axes[i],
            font_prop=devanagari_font
        )
    
    plt.tight_layout()
    # Log to W&B
    wandb.log({"connectivity_visualizations": wandb.Image(fig)})
    
    # Always save figures locally
    connectivity_fig_path = os.path.join(fig_dir, f"connectivity_visualizations_{config.language}_{run_id}.png")
    fig.savefig(connectivity_fig_path, dpi=300, bbox_inches='tight')
    print(f"Connectivity visualizations saved to {connectivity_fig_path}")
    
    # Log paths to W&B as well
    wandb.log({
        "prediction_path": output_file,
        "heatmap_fig_path": heatmap_fig_path,
        "connectivity_fig_path": connectivity_fig_path
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()