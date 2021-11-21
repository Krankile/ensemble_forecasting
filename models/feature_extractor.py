from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim, hidden_dim, dropout):
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            proj_size=self.embedding_dim,
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm1(x)
        return h_n[-1].unsqueeze(1)


class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim, hidden_dim, n_features):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            proj_size=n_features,
            batch_first=True,
        )

    def forward(self, x, lens):
        x = x.repeat(1, self.seq_len, 1)
        x = pack_padded_sequence(
            x, lens, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm1(x)

        return x


class RecurrentAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim, hidden_dim, dropout):
        super().__init__()

        self.encoder = Encoder(seq_len, n_features,
                                embedding_dim, hidden_dim, dropout)
        self.decoder = Decoder(seq_len, embedding_dim, hidden_dim, n_features)

    def forward(self, x, lens):

        x = self.encoder(x)
        x = self.decoder(x, lens)

        return x
