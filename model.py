import torch.nn as nn
import torch.nn.functional as F
import torch


class SciLSTM(nn.Module):

    def __init__(self, emb_dim=300, dropout_rate=0.2):
        super(SciLSTM, self).__init__()
        n_classes = 3
        self.hidden_size = 50
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.attn = Attention(self.hidden_size*2)
        self.classifier = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(20, n_classes)
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        attn = self.attn(h)
        out = self.classifier(attn)

        return out

class SciBertCls(nn.Module):

    def __init__(self, dropout_rate=0.2):
        super(SciBertCls, self).__init__()
        n_classes = 3
        self.dropout_rate = dropout_rate
        self.cls = nn.Sequential(
            nn.Linear(768, n_classes),
        )
        
    def forward(self, x):
        return self.cls(x)

class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = nn.Parameter(torch.FloatTensor(attention_size, 1))
        torch.nn.init.xavier_normal_(self.attention)

    def forward(self, x_in, reduction_dim=-2):
        # calculate attn weights
        attn_score = torch.matmul(x_in, self.attention).squeeze()
        # add one dimension at the end and get a distribution out of scores
        attn_distrib = F.softmax(attn_score.squeeze(), dim=-1).unsqueeze(-1)
        scored_x = x_in * attn_distrib
        weighted_sum = torch.sum(scored_x, dim=reduction_dim)
        return weighted_sum

if __name__ == '__main__':
    model = SciLSTM()
    dummy_input = torch.randn(32, 60, 300)
    out = model(dummy_input)
    print(out.shape)