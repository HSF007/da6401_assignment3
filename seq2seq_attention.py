import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
        self.v = nn.Linear(decoder_hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Reshape hidden to match encoder_outputs for concatenation
        # For GRU/RNN: hidden is [num_layers, batch_size, hidden_size]
        # For LSTM: hidden is a tuple of [h, c], each [num_layers, batch_size, hidden_size]
        
        # Handle both LSTM and GRU/RNN cases
        if isinstance(hidden, tuple):  # LSTM case
            hidden_state = hidden[0]  # Use the hidden state (h), not cell state (c)
        else:  # GRU/RNN case
            hidden_state = hidden
            
        # Take the last layer's hidden state and expand it to match encoder_outputs length
        last_hidden = hidden_state[-1].unsqueeze(1).expand(-1, src_len, -1)
        
        # Now hidden and encoder_outputs should have compatible dimensions for concatenation
        # last_hidden: [batch_size, src_len, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]
        energy = torch.tanh(self.attn(torch.cat((last_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, dropout, cell_type="GRU"):
        super(AttentionDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_size, hidden_size)
        
        if cell_type == "GRU":
            self.rnn = nn.GRU(
                embed_size + hidden_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif cell_type == "LSTM":
            self.rnn = nn.LSTM(
                embed_size + hidden_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                embed_size + hidden_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        self.fc_out = nn.Linear(hidden_size * 2 + embed_size, output_size)
    
    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        embedded = embedded.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden, a.squeeze(1)

class Seq2SeqAttention(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 embed_size, 
                 hidden_size, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dropout, 
                 cell_type="GRU"):
        super(Seq2SeqAttention, self).__init__()
        
        from seq2seq import Encoder
        
        self.encoder = Encoder(
            input_size, 
            embed_size, 
            hidden_size, 
            num_encoder_layers, 
            dropout, 
            cell_type
        )
        
        self.decoder = AttentionDecoder(
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
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        attentions = torch.zeros(batch_size, target_len, source.shape[1]).to(source.device)
        
        encoder_outputs, hidden = self.encoder(source, source_lengths)
        
        hidden = self._process_hidden_for_decoder(hidden, batch_size)
        
        decoder_input = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden, attention = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            attentions[:, t, :] = attention
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            decoder_input = target[:, t] if teacher_force else top1
        
        return outputs, attentions
    
    def _process_hidden_for_decoder(self, hidden, batch_size):
        if self.num_encoder_layers == self.num_decoder_layers:
            return hidden
        
        if self.cell_type != "LSTM":
            if self.num_encoder_layers < self.num_decoder_layers:
                additional_layers = self.num_decoder_layers - self.num_encoder_layers
                last_layer = hidden[-1:].expand(additional_layers, batch_size, self.hidden_size)
                return torch.cat([hidden, last_layer], dim=0)
            else:
                return hidden[-self.num_decoder_layers:]
        else:
            h, c = hidden
            if self.num_encoder_layers < self.num_decoder_layers:
                additional_layers = self.num_decoder_layers - self.num_encoder_layers
                last_h_layer = h[-1:].expand(additional_layers, batch_size, self.hidden_size)
                last_c_layer = c[-1:].expand(additional_layers, batch_size, self.hidden_size)
                new_h = torch.cat([h, last_h_layer], dim=0)
                new_c = torch.cat([c, last_c_layer], dim=0)
                return (new_h, new_c)
            else:
                return (h[-self.num_decoder_layers:], c[-self.num_decoder_layers:])
    
    def predict(self, source, source_len, target_vocab_size, sos_idx, eos_idx, max_len=100):
        batch_size = source.shape[0]
        
        encoder_outputs, hidden = self.encoder(source, source_len)
        
        hidden = self._process_hidden_for_decoder(hidden, batch_size)
        
        decoder_input = torch.tensor([sos_idx] * batch_size).to(source.device)
        
        predictions = []
        attention_weights = []
        
        done = [False] * batch_size
        
        for _ in range(max_len):
            output, hidden, attention = self.decoder(decoder_input, hidden, encoder_outputs)
            
            top1 = output.argmax(1)
            
            predictions.append(top1.unsqueeze(1))
            attention_weights.append(attention.unsqueeze(1))
            
            for i in range(batch_size):
                if top1[i].item() == eos_idx:
                    done[i] = True
            
            if all(done):
                break
            
            decoder_input = top1
        
        return torch.cat(predictions, dim=1), torch.cat(attention_weights, dim=1)