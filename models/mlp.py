from . import *

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        in_dim = args.in_dim
        out_dim = args.out_dim
        hidden_dim = args.hidden_dim

        self.dp = nn.Dropout(args.dropout_rate)
        self.linear_in = nn.Linear(in_dim, hidden_dim[0])
        self.relu = nn.ReLU()
        self.fc_list = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            if args.bn:
                self.fc_list.append(nn.BatchNorm1d(hidden_dim[i+1]))
        self.linear_out = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, x):
        out = self.linear_in(x)
        out = self.dp(out)
        out = self.relu(out)
        for _, layer in enumerate(self.fc_list):
            out = layer(out)
            out = self.dp(out)
            out = self.relu(out)
        out = self.linear_out(out)
        out = F.log_softmax(out, dim=1)
        return out