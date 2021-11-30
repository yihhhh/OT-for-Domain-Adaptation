class Mapping(nn.Module):
    def __init__(self, ot_plan, dim):
        super().__init__()
        self.ot_plan = ot_plan
        self.device = self.ot_plan.device
        self.map_func = nn.Sequential(nn.Linear(dim, 2*dim),
                                      nn.ReLU(),
                                      nn.Linear(2*dim, 4*dim),
                                      nn.ReLU(),
                                      nn.Linear(4*dim, 4*dim),
                                      nn.ReLU(),
                                      nn.Linear(4*dim, 2*dim),
                                      nn.ReLU(),
                                      nn.Linear(2*dim, dim)
                                      ).to(self.device)

    def reset_parameters(self):
        for module in self.map_func._modules.values():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, x):
        return self.map_func(x)

    def loss(self, x, y, xidx=None, yidx=None):
        mapped = self.map_func(x)
        distance = l2_distance(mapped, y)
        with torch.no_grad():
            plan = self.ot_plan(x, y, xidx, yidx)
        return torch.mean(plan * distance)

    def save_model(self, save_name):
        map_save_name = os.path.join(save_name, "map_func.pkl")
        torch.save(self.map_func, map_save_name)

    def load_model(self, map_model):
        self.map_func = torch.load(map_model)