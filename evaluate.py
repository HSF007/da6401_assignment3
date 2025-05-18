import torch
import wandb
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import Counter
from matplotlib.font_manager import FontProperties
import matplotlib as mpl


from config import Config
from dataloader import get_dataloader
from seq2seq import Seq2Seq
from utils import indices_to_string, create_directory

def evaluate_model(model, dataloader, dataset, device, config):
    model.eval()
    results = []
    total_correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move tensors to device
            source = batch['source'].to(device)
            source_lengths = batch['source_lengths']
            source_texts = batch['source_texts']
            target_texts = batch['target_texts']
            
            # Generate predictions
            predictions = model.predict(
                source,
                source_lengths,
                dataset.target_vocab_size,
                dataset.target_char_to_idx['< SOS >'],
                dataset.target_char_to_idx['<EOS>']
            )
            
            # Process each example in the batch
            for i in range(len(predictions)):
                pred_seq = [idx.item() for idx in predictions[i]]
                pred_str = indices_to_string(pred_seq, dataset.target_idx_to_char, dataset.target_char_to_idx['<EOS>'])
                target_str = batch['target_texts'][i]

                # Check if prediction is correct
                is_correct = pred_str == target_str
                if is_correct:
                    total_correct += 1
                
                # Store result
                results.append({
                    'source': source_texts[i],
                    'target': target_texts[i],
                    'prediction': pred_str,
                    'correct': is_correct
                })
                
            total_examples = len(predictions)
    
    # Calculate accuracy
    accuracy = total_correct / total_examples
    
    return results, accuracy

def get_best_model_from_sweep(config, device, sweep_id=None):
    """Get the best model configuration from a completed sweep"""
    
    # Initialize W&B API
    api = wandb.Api()
    
    # Get best run from the project based on validation accuracy
    try:
        if sweep_id is not None:
            sweep = api.sweep(f"{config.wandb_entity}/{config.wandb_project}/{sweep_id}")
        else:
            print('No sweep ID provided. Please provide a valid sweep ID.')
            raise ValueError("Sweep ID is required to find the best model in a sweep.")
        
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
            artifact = api.artifact(f"{config.wandb_entity}/{config.wandb_project}/model-{config.language}-{best_run.id}:latest")
            artifact_dir = artifact.download()
            best_model_path = os.path.join(artifact_dir, "best_model.pt")
            
            # If artifact download failed, try to get the file directly
            if not os.path.exists(best_model_path):
                best_model_path = best_run.file("best_model.pt").download()
            
            print(f"Downloaded best model from W&B")
        except Exception as e:
            print(f"Failed to download model from W&B: {str(e)}")
            # Look for local model with matching run ID
            model_dir = config.model_dir
            model_files = [f for f in os.listdir(model_dir) if f.startswith(f'best_model_{config.language}_{best_run.id}')]
            
            if model_files:
                best_model_path = os.path.join(model_dir, model_files[0])
                print(f"Using local model: {best_model_path}")
            else:
                # If no matching model, use the most recent local model
                model_files = [f for f in os.listdir(model_dir) if f.startswith(f'best_model_{config.language}_')]
                if model_files:
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                    best_model_path = os.path.join(model_dir, model_files[0])
                    print(f"Using most recent local model instead: {best_model_path}")
                else:
                    raise FileNotFoundError(f"No model file found for language '{config.language}'. Please train a model first.")
        
        return best_config, best_model_path
    
    except Exception as e:
        print(f"Error finding best model: {str(e)}")
        print("Looking for local model instead...")
        
        # Look for local model
        model_dir = config.model_dir
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' not found")
            
        model_files = [f for f in os.listdir(model_dir) if f.startswith(f'best_model_{config.language}_')]
        
        if model_files:
            # Use the most recent model file
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            best_model_path = os.path.join(model_dir, model_files[0])
            print(f"Using local model: {best_model_path}")
            # Keep the default config
            best_config = {
                'embed_size': config.embed_size,
                'hidden_size': config.hidden_size,
                'num_encoder_layers': config.num_encoder_layers,
                'num_decoder_layers': config.num_decoder_layers,
                'cell_type': config.cell_type,
                'dropout': config.dropout
            }
            return best_config, best_model_path
        else:
            raise FileNotFoundError(f"No model file found for language '{config.language}'. Please train a model first.")

def create_prediction_visualizations(results, config):
    """Create visualizations for the prediction results"""
    
    # Try to use a font that supports Devanagari if available
    # First, attempt to find a suitable font
    font_found = False
    
    # List of potential fonts that might support Devanagari
    potential_fonts = ['Nirmala UI', 'Mangal', 'Arial Unicode MS', 'FreeSerif', 'Noto Sans Devanagari']
    
    for font in potential_fonts:
        try:
            # Check if the font is available
            test_prop = FontProperties(family=font)
            if test_prop.get_family()[0] == font:
                # Set this font as the default
                plt.rcParams['font.family'] = font
                print(f"Using font: {font} for Devanagari support")
                font_found = True
                break
        except:
            continue
    
    if not font_found:
        print("Warning: No font with Devanagari support found. Text may not display correctly.")
        # Use a fallback approach - encode special characters
        def sanitize_text(text):
            # Replace Devanagari characters with their Unicode code point representation
            return ''.join([f"U+{ord(c):04X}" if ord(c) > 255 else c for c in text])
    else:
        # If font is found, we can use original text
        def sanitize_text(text):
            return text
    
    # Create a results DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Apply sanitization to text columns
    df['source_display'] = df['source'].apply(sanitize_text)
    df['target_display'] = df['target'].apply(sanitize_text)
    df['prediction_display'] = df['prediction'].apply(sanitize_text)
    
    # Calculate overall statistics
    accuracy = df['correct'].mean()
    total_samples = len(df)
    correct_samples_count = df['correct'].sum()
    
    # Handle possible empty groups
    correct_df = df[df['correct'] == True]
    incorrect_df = df[df['correct'] == False]
    
    # Sample safely - ensuring we don't try to sample more items than available
    correct_sample_size = min(5, len(correct_df))
    incorrect_sample_size = min(5, len(incorrect_df))
    
    # Sample from each group
    correct_samples = correct_df.sample(correct_sample_size) if correct_sample_size > 0 else pd.DataFrame()
    incorrect_samples = incorrect_df.sample(incorrect_sample_size) if incorrect_sample_size > 0 else pd.DataFrame()
    
    # Combine samples
    samples = pd.concat([correct_samples, incorrect_samples])
    
    # If we have samples to display, shuffle them
    if len(samples) > 0:
        samples = samples.sample(frac=1)  # Shuffle the combined samples
    
    # Determine the number of plots to display
    n_samples = len(samples)
    
    if n_samples == 0:
        # Handle the case where there are no samples
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No samples available for visualization", 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
    else:
        # Create a grid for the samples
        rows = min(5, max(1, (n_samples + 1) // 2))  # At least 1 row, at most 5 rows
        cols = min(2, n_samples)  # At most 2 columns
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
        
        # Handle the case of a single subplot
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        
        # Flatten axes array for easier indexing
        axes = np.array(axes).flatten()
        
        # Plot each sample
        for i, (idx, sample) in enumerate(samples.iterrows()):
            if i >= len(axes):
                break
                
            source = sample['source_display']  # Use sanitized text
            target = sample['target_display']  # Use sanitized text
            prediction = sample['prediction_display']  # Use sanitized text
            correct = sample['correct']
            
            bg_color = "#d0f0d0" if correct else "#f0d0d0"
            title_color = "green" if correct else "red"
            
            # Create a table visualization but avoid displaying directly
            # Instead, create a text display
            ax = axes[i]
            ax.axis('off')
            
            # Create text display instead of table
            text_content = f"Source: {source}\nTarget: {target}\nPrediction: {prediction}"
            
            # Display text with background color for prediction
            ax.text(0.5, 0.5, text_content, 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", 
                             facecolor=bg_color if correct else "#f0d0d0",
                             alpha=0.5))
            
            ax.set_title(f"Sample {i+1}: {'Correct' if correct else 'Incorrect'}", color=title_color)
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure with a more basic approach
    visualization_path = os.path.join(config.prediction_dir, 'prediction_grid.png')
    plt.savefig(visualization_path, bbox_inches='tight')
    plt.close()
    
    # Create error analysis plots
    
    # 1. Correct vs Incorrect counts
    plt.figure(figsize=(10, 6))
    # Fixed: Use hue parameter instead of direct palette assignment
    sns.countplot(x='correct', hue='correct', data=df, palette=["#f0d0d0", "#d0f0d0"], legend=False)
    plt.title(f"Prediction Results: {correct_samples_count} Correct / {total_samples} Total ({accuracy:.2%})")
    plt.xlabel("Prediction Correctness")
    plt.ylabel("Count")
    plt.xticks([0, 1], ["Incorrect", "Correct"])
    
    # Save the figure
    plt.tight_layout()
    error_count_path = os.path.join(config.prediction_dir, 'correct_vs_incorrect.png')
    plt.savefig(error_count_path)
    plt.close()
    
    # 2. Error analysis by string length
    plt.figure(figsize=(12, 6))
    
    # Add string length to DataFrame
    df['source_length'] = df['source'].apply(len)
    df['accuracy_by_length'] = df.groupby('source_length')['correct'].transform('mean')
    
    # Group by length
    length_stats = df.groupby('source_length').agg(
        count=('correct', 'count'),
        accuracy=('correct', 'mean')
    ).reset_index()
    
    # Create the plot - only include lengths with enough samples
    # Get lengths with at least 3 samples or 1% of the dataset
    min_count = max(3, len(df) * 0.01)
    length_stats_filtered = length_stats[length_stats['count'] >= min_count]
    
    if len(length_stats_filtered) > 0:
        # Plot accuracy by length
        plt.bar(length_stats_filtered['source_length'], length_stats_filtered['accuracy'], 
                color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Source Word Length')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Word Length')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels
        for idx, row in length_stats_filtered.iterrows():
            plt.text(row['source_length'], row['accuracy'] + 0.02, 
                    f"n={row['count']}", ha='center', fontsize=8)
        
        # Improve appearance
        plt.ylim(0, 1.1)
    else:
        # Handle case with no data
        plt.text(0.5, 0.5, "Insufficient data for length analysis", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # Save the figure
    length_analysis_path = os.path.join(config.prediction_dir, 'accuracy_by_length.png')
    plt.savefig(length_analysis_path)
    plt.close()
    
    # 3. Character-level error analysis
    incorrect_df = df[df['correct'] == False].copy()  # Fixed: Create a copy to avoid SettingWithCopyWarning
    
    char_error_path = None
    # Check if there are any incorrect predictions for analysis
    if len(incorrect_df) > 0:
        # Function to find character-level differences - use sanitized text
        def get_char_differences(row):
            target = row['target']
            pred = row['prediction']
            
            # Simple character error analysis
            min_len = min(len(target), len(pred))
            errors = []
            
            for i in range(min_len):
                if target[i] != pred[i]:
                    errors.append((i, target[i], pred[i]))
            
            # Handle length differences
            if len(target) > len(pred):
                for i in range(min_len, len(target)):
                    errors.append((i, target[i], ""))
            elif len(pred) > len(target):
                for i in range(min_len, len(pred)):
                    errors.append((i, "", pred[i]))
                    
            return errors
        
        # Apply function to get character differences
        incorrect_df['char_errors'] = incorrect_df.apply(get_char_differences, axis=1)
        
        # Extract all character errors
        all_errors = []
        for errors in incorrect_df['char_errors']:
            for pos, target_char, pred_char in errors:
                # Use sanitized characters for display
                t_char = sanitize_text(target_char)
                p_char = sanitize_text(pred_char)
                all_errors.append((t_char, p_char))
        
        # Count pair occurrences
        error_counts = Counter(all_errors)
        
        # Get the top 15 most common errors
        top_errors = error_counts.most_common(15)
        
        if top_errors:
            # Create a DataFrame for visualization
            error_df = pd.DataFrame(top_errors, columns=['char_pair', 'count'])
            error_df['target_char'] = error_df['char_pair'].apply(lambda x: x[0] if x[0] else '(empty)')  # Fixed: Use "(empty)" instead of "∅"
            error_df['pred_char'] = error_df['char_pair'].apply(lambda x: x[1] if x[1] else '(empty)')  # Fixed: Use "(empty)" instead of "∅"
            error_df['pair_label'] = error_df.apply(lambda row: f"{row['target_char']} -> {row['pred_char']}", axis=1)  # Fixed: Use "->" instead of "→"
            
            plt.figure(figsize=(14, 8))
            
            # Use a horizontal bar chart for better readability
            bars = plt.barh(error_df['pair_label'], error_df['count'], color='salmon', edgecolor='darkred')
            plt.xlabel('Count')
            plt.ylabel('Character Error (Target -> Prediction)')  # Fixed: Use "->" instead of "→"
            plt.title('Most Common Character-Level Errors')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{width}', va='center', fontsize=9)
            
            plt.tight_layout()
            
            # Save the figure
            char_error_path = os.path.join(config.prediction_dir, 'character_errors.png')
            plt.savefig(char_error_path)
            plt.close()
    
    return {
        'prediction_grid': visualization_path,
        'error_count': error_count_path,
        'length_analysis': length_analysis_path,
        'char_error': char_error_path
    }

def main():
    parser = argparse.ArgumentParser(description='Run evaluation for best model in a sweep')
    parser.add_argument('--sweep_id', type=str, help='The sweep ID to use (if already created)')

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    sweep_id = args.sweep_id

    # Create vanilla prediction directory specifically for Question 4
    vanilla_pred_dir = os.path.join(os.path.dirname(config.prediction_dir), "predictions_vanilla")
    if not os.path.exists(vanilla_pred_dir):
        os.makedirs(vanilla_pred_dir, exist_ok=True)
    
    # Temporarily override the prediction directory
    original_pred_dir = config.prediction_dir
    config.prediction_dir = vanilla_pred_dir
    
    # Initialize W&B for evaluation
    wandb.init(project=config.wandb_project, entity=config.wandb_entity, job_type="evaluation", 
               name=f"vanilla_eval_{config.language}")
    
    # Log the language being evaluated
    wandb.config.update({"language": config.language})
    
    # Load test data
    test_loader, test_dataset = get_dataloader(
        config.test_file, 
        config.batch_size, 
        shuffle=False
    )
    
    # Get the best model from sweep
    best_config, best_model_path = get_best_model_from_sweep(config, device, sweep_id)

    # Load the model checkpoint to get the correct vocabulary size
    checkpoint = torch.load(best_model_path, map_location=device)

    # Extract the vocabulary size from the model checkpoint
    output_size = checkpoint['decoder.fc_out.bias'].size(0)  # This should be 66 based on the error

    # Create model with the best configuration and correct vocabulary size
    model = Seq2Seq(
        input_size=test_dataset.source_vocab_size,
        output_size=output_size,  # Use the output size from the checkpoint instead
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
    
    # Create prediction directory
    create_directory(config.prediction_dir)
    
    # Evaluate model
    results, accuracy = evaluate_model(model, test_loader, test_dataset, device, config)
    
    # Create visualizations
    visualization_paths = create_prediction_visualizations(results, config)
    
    # Save results
    output_file = os.path.join(config.prediction_dir, f"predictions_{config.language}_{wandb.run.id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Log accuracy to W&B
    wandb.log({'test_accuracy': accuracy, 'language': config.language})
    
    print(f'Test Accuracy for {config.language}: {accuracy:.4f}')
    print(f'Predictions saved to {output_file}')
    
    # Create a table for W&B
    columns = ["source", "target", "prediction", "correct"]
    data = [[r["source"], r["target"], r["prediction"], r["correct"]] for r in results]
    table = wandb.Table(columns=columns, data=data)
    wandb.log({"predictions": table})
    
    # Log the visualization as W&B images
    for name, path in visualization_paths.items():
        if path and os.path.exists(path):
            image = wandb.Image(path)
            wandb.log({name: image})
            print(f"logged {name} to W&B from {path}")
    
    # Log some sample predictions
    samples = results[:10]  # First 10 samples
    sample_data = "\n".join([f"Source: {s['source']}, Target: {s['target']}, Prediction: {s['prediction']}, Correct: {s['correct']}" for s in samples])
    wandb.log({"samples": wandb.Html(sample_data)})
    
    # Analyze error patterns
    error_analysis = analyze_errors(results)
    wandb.log({"error_analysis": wandb.Html(error_analysis)})
    
    # Restore original prediction directory
    config.prediction_dir = original_pred_dir
    
    wandb.finish()

def analyze_errors(results):
    """Analyze common error patterns in the predictions"""
    
    df = pd.DataFrame(results)
    
    # Filter for incorrect predictions
    incorrect = df[df['correct'] == False].copy()  # Fixed: Create a copy to avoid SettingWithCopyWarning
    
    if len(incorrect) == 0:
        return "<p>No errors to analyze!</p>"
    
    # Error patterns analysis
    analysis = "<h3>Error Analysis</h3>"
    
    # 1. Length differences
    incorrect.loc[:, 'target_len'] = incorrect['target'].apply(len)  # Fixed: Use .loc to avoid SettingWithCopyWarning
    incorrect.loc[:, 'pred_len'] = incorrect['prediction'].apply(len)  # Fixed: Use .loc to avoid SettingWithCopyWarning
    incorrect.loc[:, 'len_diff'] = incorrect['pred_len'] - incorrect['target_len']  # Fixed: Use .loc to avoid SettingWithCopyWarning
    
    len_diff_counts = incorrect['len_diff'].value_counts().head(5)
    len_diff_html = "<h4>Length Differences in Incorrect Predictions</h4>"
    len_diff_html += "<ul>"
    
    for diff, count in len_diff_counts.items():
        len_diff_html += f"<li>Difference of {diff} characters: {count} occurrences ({(count/len(incorrect))*100:.1f}%)</li>"
    
    len_diff_html += "</ul>"
    analysis += len_diff_html
    
    # 2. Example error cases for inspection
    analysis += "<h4>Sample Error Cases</h4>"
    analysis += "<table border='1' style='border-collapse: collapse;'>"
    analysis += "<tr><th>Source</th><th>Target</th><th>Prediction</th><th>Analysis</th></tr>"
    
    # Select diverse error examples
    error_samples = []
    
    # Shorter prediction
    shorter_pred = incorrect[incorrect['len_diff'] < 0].head(1)
    if not shorter_pred.empty:
        error_samples.append(shorter_pred.iloc[0])
    
    # Longer prediction
    longer_pred = incorrect[incorrect['len_diff'] > 0].head(1)
    if not longer_pred.empty:
        error_samples.append(longer_pred.iloc[0])
    
    # Same length but incorrect
    same_len = incorrect[incorrect['len_diff'] == 0].head(1)
    if not same_len.empty:
        error_samples.append(same_len.iloc[0])
    
    # Add more examples if needed
    if len(error_samples) < 3:
        more_examples = incorrect.head(3 - len(error_samples))
        error_samples.extend(more_examples.to_dict('records'))
    
    for sample in error_samples:
        source = sample['source']
        target = sample['target']
        prediction = sample['prediction']
        
        # Simple error analysis
        analysis_text = ""
        
        if len(prediction) < len(target):
            analysis_text = f"Missing characters (prediction too short by {len(target) - len(prediction)} chars)"
        elif len(prediction) > len(target):
            analysis_text = f"Extra characters (prediction too long by {len(prediction) - len(target)} chars)"
        else:
            # Find positions of difference
            diff_positions = []
            for i, (t, p) in enumerate(zip(target, prediction)):
                if t != p:
                    diff_positions.append(i)
            analysis_text = f"Character mismatch at positions: {diff_positions}"
        
        analysis += f"<tr><td>{source}</td><td>{target}</td><td>{prediction}</td><td>{analysis_text}</td></tr>"
    
    analysis += "</table>"
    
    return analysis

if __name__ == "__main__":
    main()