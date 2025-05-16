import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout, cell_type="GRU"):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
        if cell_type == "GRU":
            self.rnn = nn.GRU(
                embed_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif cell_type == "LSTM":
            self.rnn = nn.LSTM(
                embed_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # Default to RNN
            self.rnn = nn.RNN(
                embed_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        self.cell_type = cell_type
    
    def forward(self, x, lengths):
        # x shape: (batch_size, seq_length)
        embedded = self.dropout(self.embedding(x))
        
        # Pack padded batch of sequences for RNN
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Forward propagate RNN
        outputs, hidden = self.rnn(packed)
        
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, dropout, cell_type="GRU"):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
        if cell_type == "GRU":
            self.rnn = nn.GRU(
                embed_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif cell_type == "LSTM":
            self.rnn = nn.LSTM(
                embed_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # Default to RNN
            self.rnn = nn.RNN(
                embed_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        # x shape: (batch_size, 1)
        embedded = self.dropout(self.embedding(x))
        
        # Forward propagate through RNN
        output, hidden = self.rnn(embedded, hidden)
        
        # Compute output
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 embed_size, 
                 hidden_size, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dropout, 
                 cell_type="GRU"):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(
            input_size, 
            embed_size, 
            hidden_size, 
            num_encoder_layers, 
            dropout, 
            cell_type
        )
        
        self.decoder = Decoder(
            output_size, 
            embed_size, 
            hidden_size, 
            num_decoder_layers, 
            dropout, 
            cell_type
        )
        
        self.cell_type = cell_type
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
    
    def forward(self, source, target, source_lengths, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        
        # Encode source sequences
        encoder_outputs, hidden = self.encoder(source, source_lengths)
        
        # Process hidden state if encoder and decoder layers differ
        hidden = self._process_hidden_for_decoder(hidden, batch_size)
        
        # First input to the decoder is the < SOS > token
        decoder_input = target[:, 0].unsqueeze(1)
        
        # Decode one character at a time
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output
            
            # Teacher forcing: decide whether to use the predicted or actual target as next input
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # If teacher forcing, use actual next token as input; otherwise use predicted token
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def _process_hidden_for_decoder(self, hidden, batch_size):
        """Process the encoder hidden state for the decoder"""
        if self.num_encoder_layers == self.num_decoder_layers:
            return hidden
        
        # For GRU and RNN
        if self.cell_type != "LSTM":
            if self.num_encoder_layers < self.num_decoder_layers:
                # Duplicate last layer to match decoder layers
                additional_layers = self.num_decoder_layers - self.num_encoder_layers
                last_layer = hidden[-1:].expand(additional_layers, batch_size, self.hidden_size)
                return torch.cat([hidden, last_layer], dim=0)
            else:
                # Use only the last n layers
                return hidden[-self.num_decoder_layers:]
        else:
            # For LSTM (hidden is a tuple of (h_0, c_0))
            h, c = hidden
            if self.num_encoder_layers < self.num_decoder_layers:
                # Duplicate last layer to match decoder layers
                additional_layers = self.num_decoder_layers - self.num_encoder_layers
                last_h_layer = h[-1:].expand(additional_layers, batch_size, self.hidden_size)
                last_c_layer = c[-1:].expand(additional_layers, batch_size, self.hidden_size)
                new_h = torch.cat([h, last_h_layer], dim=0)
                new_c = torch.cat([c, last_c_layer], dim=0)
                return (new_h, new_c)
            else:
                # Use only the last n layers
                return (h[-self.num_decoder_layers:], c[-self.num_decoder_layers:])
    
    def predict(self, source, source_len, target_vocab_size, sos_idx, eos_idx, max_len=100):
        batch_size = source.shape[0]
        
        # Encode source sequences
        encoder_outputs, hidden = self.encoder(source, source_len)
        
        # Process hidden state if encoder and decoder layers differ
        hidden = self._process_hidden_for_decoder(hidden, batch_size)
        
        # First input to the decoder is the < SOS > token
        decoder_input = torch.tensor([[sos_idx]] * batch_size).to(source.device)
        
        # Lists to store predicted outputs
        predictions = []
        
        # Flag to indicate if decoding is complete
        done = [False] * batch_size
        
        # Decode one character at a time
        for _ in range(max_len):
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Store predicted token
            predictions.append(top1.unsqueeze(1))
            
            # Check if all sequences have reached EOS
            for i in range(batch_size):
                if top1[i].item() == eos_idx:
                    done[i] = True
            
            if all(done):
                break
            
            # Use predicted token as next input
            decoder_input = top1.unsqueeze(1)
        
        # Concatenate predictions along the sequence dimension
        return torch.cat(predictions, dim=1)