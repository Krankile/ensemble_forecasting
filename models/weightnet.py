import torch
from torch import nn

from ..utils import activations


class WeightNetV3(nn.Module):
    def __init__(
        self,
        num_cont,
        out_size,
        n_hidden,
        hidden_dim,
        dropout,
        bn,
        activation,
        emb_dims,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.num_embs = sum([y for x, y in emb_dims])
        self.num_cont = num_cont

        layers = [nn.Linear(self.num_embs + self.num_cont, hidden_dim)]

        self.first_bn = nn.BatchNorm1d(self.num_cont)

        for i in range(n_hidden):
            layers.extend(
                [nn.Dropout(p=dropout)] + [nn.BatchNorm1d(hidden_dim)]
                if bn
                else [] + [activations[activation]()]
            )
            if i == (n_hidden - 1):
                layers.append(nn.Linear(hidden_dim, out_size))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, cat, cont):
        x = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, 1)

        cont = self.first_bn(cont)

        x = torch.cat([x, cont], 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class WeightNetV4(nn.Module):
    def __init__(
        self,
        num_cont,
        out_size,
        n_hidden,
        hidden_dim,
        dropout,
        bn,
        activation,
        emb_dims,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.num_embs = sum([y for x, y in emb_dims])
        self.num_cont = num_cont

        layers = [nn.Linear(self.num_embs + self.num_cont, hidden_dim)]

        self.first_cont = nn.BatchNorm1d(self.num_cont) if bn else lambda x: x

        for i in range(n_hidden):
            layers.extend(
                [nn.Dropout(p=dropout)] + [nn.BatchNorm1d(hidden_dim)]
                if bn
                else [] + [activations[activation]()]
            )
            if i == (n_hidden - 1):
                layers.append(nn.Linear(hidden_dim, out_size))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, cat, cont):
        x = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, 1)

        cont = self.first_cont(cont)

        x = torch.cat([x, cont], 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


weightnets = {
    "WeightNetV3": WeightNetV3,
    "WeightNetV4": WeightNetV4,
}
