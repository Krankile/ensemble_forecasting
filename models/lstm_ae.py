from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderV2(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim, hidden_dim, dropout):
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.drop1 = nn.Dropout(p=dropout, inplace=False)

        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x, (_, _) = self.lstm1(x)
        x, lens = pad_packed_sequence(
            x, batch_first=True, total_length=self.seq_len)
        x = self.drop1(x)
        x = pack_padded_sequence(
            x, lens, batch_first=True, enforce_sorted=False)
        x, (hidden_n, _) = self.lstm2(x)

        return hidden_n.reshape((-1, self.n_features, self.embedding_dim)), lens


class DecoderV2(nn.Module):

    def __init__(self, seq_len, input_dim, hidden_dim, n_features):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, lens):
        x = x.repeat(1, self.seq_len, 1)

        x = pack_padded_sequence(
            x, lens, batch_first=True, enforce_sorted=False)

        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        x, lens = pad_packed_sequence(
            x, batch_first=True, total_length=self.seq_len)
        x = x.reshape((-1, self.seq_len, self.hidden_dim))

        return self.output_layer(x)


class RecurrentAutoencoderV2(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim, hidden_dim, dropout):
        super().__init__()

        self.encoder = EncoderV2(
            seq_len, n_features, embedding_dim, hidden_dim, dropout)
        self.decoder = DecoderV2(
            seq_len, embedding_dim, hidden_dim, n_features)

    def forward(self, x):

        x, lens = self.encoder(x)
        x = self.decoder(x, lens)

        return x


class EncoderV3(nn.Module):

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


class DecoderV3(nn.Module):

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


class RecurrentAutoencoderV3(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim, hidden_dim, dropout):
        super().__init__()

        self.encoder = EncoderV3(seq_len, n_features,
                                 embedding_dim, hidden_dim, dropout)
        self.decoder = DecoderV3(
            seq_len, embedding_dim, hidden_dim, n_features)

    def forward(self, x, lens):

        x = self.encoder(x)
        x = self.decoder(x, lens)

        return x


model_versions = {
    "RecurrentAutoencoderV2": RecurrentAutoencoderV2,
    "RecurrentAutoencoderV3": RecurrentAutoencoderV3,
}