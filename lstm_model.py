import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)
        
    def forward(self, rnn_output):
        # rnn_output: (batch_size, seq_len, hidden_size)
        
        # Calculate attention energies
        # energy: (batch_size, seq_len, 1)
        energy = torch.tanh(self.attn(rnn_output))
        
        # Calculate attention weights
        # weights: (batch_size, seq_len, 1)
        weights = torch.softmax(energy, dim=1)
        
        # Weighted sum of rnn_outputs
        # context: (batch_size, 1, hidden_size) -> (batch_size, hidden_size)
        context = torch.bmm(weights.transpose(1, 2), rnn_output).squeeze(1)
        
        return context, weights

class MicroExpressionLSTM(nn.Module):
    def __init__(self, input_size=520, hidden_size=64, num_layers=2, num_classes=7):
        """
        Args:
            input_size: Number of features per frame (512 FaceNet + 8 Emotion Scores = 520)
            hidden_size: Number of features in the hidden state
            num_layers: Number of recurrent layers
            num_classes: Number of output classes
        """
        super(MicroExpressionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq, feature)
        # Add dropout only if num_layers > 1, but we can also use manual dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=False, dropout=0.5 if num_layers > 1 else 0)
        
        # Attention Layer
        self.attention = Attention(hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        
        # Apply Attention
        context_vector, attn_weights = self.attention(lstm_out)
        
        # Apply Dropout to the context vector before FC
        out = self.dropout(context_vector)
        
        # Decode the context vector
        out = self.fc(out)
        
        return out, attn_weights
