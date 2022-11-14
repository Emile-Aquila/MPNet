import torch.nn as nn


class PlannerNetwork(nn.Module):  # Planner Network
    def __init__(self, input_size, output_size):
        super(PlannerNetwork, self).__init__()
        self.models = nn.Sequential(
            nn.Linear(input_size, 1024), nn.PReLU(), nn.Dropout(),
            nn.Linear(1024, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 64), nn.PReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        y = self.models(x)
        return y


class EncoderNetwork(nn.Module):  # Encoder Network
    def __init__(self, input_size, output_size):
        super(EncoderNetwork, self).__init__()
        self.models = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        y = self.models(x)
        return y


class DecoderNetwork(nn.Module):
    def __init__(self, encoder: EncoderNetwork):
        super(DecoderNetwork, self).__init__()
        self.models = nn.Sequential()
        for model in encoder.models:
            if isinstance(model, nn.Linear):
                self.models.append(nn.Linear(model.out_features, model.in_features))
            elif isinstance(model, nn.PReLU):
                self.models.append(nn.PReLU())

    def forward(self, x):
        y = self.models(x)
        return y


