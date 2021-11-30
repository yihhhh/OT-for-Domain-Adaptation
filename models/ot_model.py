class OTPlan(nn.Module):
    def __init__(self, source_type='discrete',
                 target_type = 'discrete',
                 source_dim = None,
                 target_dim = None,
                 source_length = None,
                 target_length = None,
                 alpha = 0.1,
                 regularization = 'entropy',
                 device = 'cpu'
                 ):
        super().__init__()
        self.source_type = source_type
        self.device = device

        if source_type == 'discrete':
            assert isinstance(source_length, int)
            self.u = DiscretePotential(source_length).to(device)
        elif source_type == 'continuous':
            assert isinstance(source_dim, int)
            self.u = ContinuousPotential(source_dim).to(device)
        self.target_type = target_type
        if target_type == 'discrete':
            assert isinstance(target_length, int)
            self.v = DiscretePotential(target_length).to(device)
        elif target_type == 'continuous':
            assert isinstance(target_dim, int)
            self.v = ContinuousPotential(target_dim).to(device)
        self.alpha = alpha

        assert regularization in ['entropy', 'l2'], ValueError
        self.regularization = regularization
        self.reset_parameters()

    def reset_parameters(self):
        self.u.reset_parameters()
        self.v.reset_parameters()

    def _get_uv(self, x, y, xidx=None, yidx=None):
        if self.source_type == 'discrete':
            u = self.u(xidx)
        else:
            u = self.u(x)
        if self.target_type == 'discrete':
            v = self.v(yidx)
        else:
            v = self.v(y)
        return u, v

    def loss(self, x, y, xidx=None, yidx=None):
        K = torch.sqrt(l2_distance(x, y))
        u, v = self._get_uv(x, y, xidx, yidx)

        if self.regularization == 'entropy':
            reg = - self.alpha * torch.exp((u[:, None] + v[None, :] - K) / self.alpha)
        else:
            reg = - torch.clamp((u[:, None] + v[None, :] - K),
                                min=0) ** 2 / 4 / self.alpha
        return - torch.mean(u[:, None] + v[None, :] + reg)

    def forward(self, x, y, xidx=None, yidx=None):
        K = torch.sqrt(l2_distance(x, y))
        u, v = self._get_uv(x, y, xidx, yidx)
        if self.regularization == 'entropy':
            return torch.exp((u[:, None] + v[None, :] - K) / self.alpha)
        else:
            return torch.clamp((u[:, None] + v[None, :] - K),
                               min=0) / (2 * self.alpha)

    def save_model(self, save_name):
        u_save_name = os.path.join(save_name, "u.pkl")
        v_save_name = os.path.join(save_name, "v.pkl")
        torch.save(self.u, u_save_name)
        torch.save(self.v, v_save_name)

    def load_model(self, u_model, v_model):
        self.u = torch.load(u_model)
        self.v = torch.load(v_model)


class DiscretePotential(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.u = Parameter(torch.empty(length))
        self.reset_parameters()

    def reset_parameters(self):
        self.u.data.zero_()

    def forward(self, idx):
        return self.u[idx]

class ContinuousPotential(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Sequential(nn.Linear(dim, 2*dim),
                               nn.ReLU(),
                               nn.Linear(2*dim, 4*dim),
                               nn.ReLU(),
                               nn.Linear(4*dim, 2*dim),
                               nn.ReLU(),
                               nn.Linear(2*dim, 1)
                               )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.u._modules.values():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, x):
        return self.u(x)[:, 0]
