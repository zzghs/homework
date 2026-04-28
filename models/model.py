import torch
from torch import nn


class PoetryGenerator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.2,
        model_type="lstm",
    ):
        super().__init__()
        self.model_type = model_type.lower()
        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if self.model_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.model_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.recurrent.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, input_ids, hidden_state=None):
        embeddings = self.embedding(input_ids)
        outputs, hidden_state = self.recurrent(embeddings, hidden_state)
        logits = self.output_layer(outputs)
        return logits, hidden_state
