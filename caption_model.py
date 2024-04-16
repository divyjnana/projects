import torch
import torch.nn as nn
class CaptionGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(CaptionGenerationModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer to map LSTM output to vocabulary space
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        """
        Forward pass of the caption generation model.

        Args:
            features: A tensor of preprocessed video features (e.g., batch of sequences of frame features).
            captions: A tensor of ground truth captions (for teacher forcing during training, optional).

        Returns:
            A tensor of predicted caption logits (output before softmax).
        """

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.lstm.num_layers, features.size(0), self.lstm.hidden_size).to(features.device)
        c0 = torch.zeros(self.lstm.num_layers, features.size(0), self.lstm.hidden_size).to(features.device)

        # Embedding captions (optional for teacher forcing)
        if captions is not None:
            embeddings = self.embedding(captions)
        else:
            embeddings = None

        # Concatenate features and embeddings (if applicable)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1) if captions is not None else features.unsqueeze(1)

        # LSTM
        lstm_out, _ = self.lstm(inputs, (h0, c0))

        # Fully connected layer
        outputs = self.fc(lstm_out)

        # Prediction using softmax (optional for teacher forcing or inference)
        # You can uncomment this line for caption generation during inference
        # predicted_captions = torch.nn.functional.softmax(outputs, dim=2)

        return outputs  # Return logits before softmax for flexibility
