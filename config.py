import os

class Config:
    # Data parameters
    language = os.environ.get("TRANSLITERATION_LANGUAGE", "hi")  # Get language from env var or default to "hi"
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    data_dir = os.path.join(base_dir, "dakshina_dataset_v1.0", f"{language}", "lexicons")  # Use absolute path
    
    # Check if data directory exists, if not, try to find it elsewhere
    if not os.path.exists(data_dir):
        # Try to find the dakshina dataset in parent directories
        parent_dir = os.path.dirname(base_dir)
        alt_data_dir = os.path.join(parent_dir, "dakshina_dataset_v1.0", f"{language}", "lexicons")
        if os.path.exists(alt_data_dir):
            data_dir = alt_data_dir
    
    train_file = os.path.join(data_dir, f"{language}.translit.sampled.train.tsv")
    val_file = os.path.join(data_dir, f"{language}.translit.sampled.dev.tsv")
    test_file = os.path.join(data_dir, f"{language}.translit.sampled.test.tsv")
    
    # Model parameters (default values)
    embed_size = 64
    hidden_size = 128
    num_encoder_layers = 1
    num_decoder_layers = 1
    dropout = 0.2
    cell_type = "GRU"  # Options: RNN, LSTM, GRU
    
    # Training parameters
    batch_size = 64
    epochs = 20
    learning_rate = 0.001
    teacher_forcing_ratio = 0.5
    
    # Decoding parameters
    beam_size = 1  # 1 for greedy decoding
    
    # W&B parameters
    wandb_project = "dakshina_transliteration"
    wandb_entity = "da24m008-iit-madras"  # Set your W&B username
    
    # Save directories
    model_dir = os.path.join(base_dir, "saved_models", language)
    # Create directory for saving model if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    prediction_dir = os.path.join(base_dir, "predictions", language)
    # Create directory for saving prediction if it doesn't exist
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    
    def __init__(self):
        # Verify paths and print info
        print(f"Base directory: {self.base_dir}")
        print(f"Language: {self.language}")
        print(f"Data directory: {self.data_dir}")
        print(f"Train file exists: {os.path.exists(self.train_file)}")
        print(f"Val file exists: {os.path.exists(self.val_file)}")
        print(f"Test file exists: {os.path.exists(self.test_file)}")
        
        # Print sample data from files if they exist
        self._print_sample_data(self.train_file, "train")
        self._print_sample_data(self.val_file, "validation")
        self._print_sample_data(self.test_file, "test")
    
    def _print_sample_data(self, file_path, name):
        """Print a sample of data from the file for debugging"""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:3]  # Get first 3 lines
                    if lines:
                        print(f"\nSample {name} data:")
                        for line in lines:
                            print(f"  {line.strip()}")
                        print(f"  ... (total lines: {sum(1 for _ in open(file_path, 'r', encoding='utf-8'))})")
                    else:
                        print(f"\nWarning: {name} file exists but is empty: {file_path}")
            except Exception as e:
                print(f"\nError reading {name} file: {str(e)}")