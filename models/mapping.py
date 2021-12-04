from . import *

class Mapping(nn.Module):
    def __init__(self, ot_plan, dim, hidden_size, device):
        super().__init__()
        self.ot_plan = ot_plan
        self.device = device
        self.hidden_size = [dim] + hidden_size
        layers = []
        for i in range(len(self.hidden_size) - 1):
            layers.append(nn.Linear(self.hidden_size[i], self.hidden_size[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_size[-1], dim))
        self.map_func = torch.nn.Sequential(*layers).to(device)

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
        map_save_name = os.path.join(save_name, "mapping.pkl")
        torch.save(self.map_func, map_save_name)

    def load_model(self, load_name):
        map_load_name = os.path.join(load_name, "mapping.pkl")
        self.map_func = torch.load(map_load_name)