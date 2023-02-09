import hydra
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig
from torch.autograd import Variable

from models import EncoderNetwork, DecoderNetwork


class CAE:
    mse: nn.MSELoss
    encoder: EncoderNetwork
    decoder: DecoderNetwork
    dev: torch.device

    def __init__(self, input_size: int, output_size: int) -> None:
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mse = nn.MSELoss(size_average=True)
        self.encoder = EncoderNetwork(input_size, output_size).to(self.dev)
        self.decoder = DecoderNetwork(output_size, input_size).to(self.dev)

    def loss_function(self, xs, xs_hat, model_weights, lamd):
        loss = self.mse(xs, xs_hat)
        reconstruction_loss = torch.sum(Variable(model_weights) ** 2, dim=1).sum()  # TODO : 要確認
        return loss + reconstruction_loss * lamd

    def encode(self, x) -> torch.Tensor:
        y = self.encoder(x)
        return x


def load_training_data(path_data_path: str, pc_data_path: str) -> tuple[dict, dict]:
    with open(path_data_path) as file:
        path_data = yaml.load(file, yaml.Loader)
    with open(pc_data_path) as file:
        pc_data = yaml.load(file, yaml.Loader)
    return path_data, pc_data


def load_dataset(path_data_path: str, pc_data_path: str) -> tuple[list, list]:
    point_clouds, path_id_tuples = [], []
    path_data, pc_data = load_training_data(path_data_path, pc_data_path)
    for map_id in path_data.keys():
        point_clouds.append(pc_data[map_id])
        for path in path_data[map_id]:
            path_id_tuples.append((path, map_id))
    return point_clouds, path_id_tuples


def train():
    pass


def main():
    path_data, pc_data = load_training_data("./DataGeneration/id_path_data.yaml", "./DataGeneration/id_pc_data.yaml")
    point_clouds, path_id_tuples = load_dataset("./DataGeneration/id_path_data.yaml", "./DataGeneration/id_pc_data.yaml")

    print(len(point_clouds))
    print(path_id_tuples)


if __name__ == '__main__':
    main()
