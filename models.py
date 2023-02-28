import torch
import torch.nn as nn


class PlannerNetwork(nn.Module):  # Planner Network
    def __init__(self, input_size: int, output_size: int) -> None:
        super(PlannerNetwork, self).__init__()
        self.models = nn.Sequential(
            nn.Linear(input_size, 1024), nn.PReLU(), nn.Dropout(),
            nn.Linear(1024, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 64), nn.PReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor, x_target: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        tmp = torch.cat((x, x_target, z), dim=0)
        y = self.models(tmp)
        return y


class EncoderNetwork(nn.Module):  # Encoder Network
    def __init__(self, input_size: int, output_size: int) -> None:
        super(EncoderNetwork, self).__init__()
        self.models = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.PReLU(),
            # nn.Linear(512, 512),
            # nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.models(x)
        return y


class DecoderNetwork(nn.Module):
    def __init__(self, encoder: EncoderNetwork):
        super(DecoderNetwork, self).__init__()
        self.models = nn.Sequential()
        for model in reversed(encoder.models):
            if isinstance(model, nn.Linear):
                self.models.append(nn.Linear(model.out_features, model.in_features))
            elif isinstance(model, nn.PReLU):
                self.models.append(nn.PReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.models(x)
        return y


def get_model_weights_sum(sequential: nn.Sequential, norm_l: int, dev: torch.device) -> torch.Tensor:
    weight_sum = torch.tensor(0.0, requires_grad=True).to(dev)
    for w in sequential.parameters():
        weight_sum = weight_sum + torch.norm(w, norm_l)
    return weight_sum
