import hydra
import torch
import torch.nn as nn
import numpy as np
import yaml
from omegaconf import DictConfig
from torch.autograd import Variable
from tqdm import tqdm

from models import EncoderNetwork, DecoderNetwork
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def loss_function(self, xs, xs_hat, lamd=1e-3):
        loss = self.mse(xs, xs_hat)
        print(self.encoder.state_dict().keys())
        model_weights = self.encoder.state_dict()[""]  # TODO : 上のkeyの結果に合わせて適切な引数を入れる
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
        point_clouds.append(np.array(pc_data[map_id]))  # 各point_cloudはnp.array
        for path in path_data[map_id]:
            path_id_tuples.append((path, map_id))
    return point_clouds, path_id_tuples


@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
def train(conf: DictConfig):
    cae = CAE(input_size=10, output_size=10)  # TODO : input, outputを後で合わせる
    params = list(cae.encoder.parameters()) + list(cae.decoder.parameters())
    optim = torch.optim.Adagrad(params)

    point_clouds, _ = load_dataset("./DataGeneration/id_path_data.yaml", "./DataGeneration/id_pc_data.yaml")
    batch_size = conf.CAEParams.batch_size

    for epoch in tqdm(conf.CAEParams.epoch_num):
        for i in range(0, len(point_clouds), batch_size):
            cae.decoder.zero_grad()
            cae.encoder.zero_grad()
            input_clouds_pre: np.array = point_clouds[i:min(i+batch_size, len(point_clouds)-1)]
            input_clouds: torch.Tensor = torch.from_numpy(input_clouds_pre).to(dev)

            latents = cae.encode(input_clouds)
            outputs = cae.decoder(latents)
            loss = cae.loss_function(xs=point_clouds, xs_hat=outputs)

            loss.backward()
            optim.step()
            print(loss)





def main():
    point_clouds, path_id_tuples = load_dataset("./DataGeneration/id_path_data.yaml", "./DataGeneration/id_pc_data.yaml")

    print(point_clouds[0])
    print(point_clouds[0].shape)
    # print(path_id_tuples)


if __name__ == '__main__':
    main()
