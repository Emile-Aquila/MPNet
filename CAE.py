import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models import EncoderNetwork, DecoderNetwork, get_model_weights_sum
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
        self.decoder = DecoderNetwork(self.encoder).to(self.dev)

    def loss_function(self, xs: torch.Tensor, xs_hat: torch.Tensor, lamd=1e-3) -> torch.Tensor:
        loss = self.mse(xs, xs_hat)
        reconstruction_loss = get_model_weights_sum(self.encoder.models, 2, self.dev)
        return loss + reconstruction_loss.mul_(lamd)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x)
        return y


def load_training_data(path_data_path: str, pc_data_path: str) -> tuple[dict, dict]:
    with open(path_data_path) as file:
        path_data = yaml.load(file, yaml.Loader)
    with open(pc_data_path) as file:
        pc_data = yaml.load(file, yaml.Loader)
    return path_data, pc_data


def load_dataset(path_data_path: str, pc_data_path: str) -> tuple[np.array, list]:
    # TODO: 評価用のデータセットを分けて準備して、過学習してないか評価する
    point_clouds, path_id_tuples = [], []
    path_data, pc_data = load_training_data(path_data_path, pc_data_path)
    for map_id in path_data.keys():
        point_clouds.append(np.array(pc_data[map_id]).flatten())  # 各point_cloudはnp.array
        for path in path_data[map_id]:
            path_id_tuples.append((np.array(path), map_id))
    return np.array(point_clouds), path_id_tuples


@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
def train(conf: DictConfig):
    input_size = conf.TrainingDataParams.point_cloud_num * conf.CAEParams.coordinates_dim
    output_size = conf.CAEParams.latent_space_size
    cae = CAE(input_size=input_size, output_size=output_size)
    print("Encoder Network: {}".format(cae.encoder.models))
    print("Decoder Network: {}".format(cae.decoder.models))

    writer = SummaryWriter(log_dir="./logs/CAE")
    writer.add_graph(cae.encoder, torch.zeros(input_size))

    params = list(cae.encoder.parameters()) + list(cae.decoder.parameters())
    optim = torch.optim.Adagrad(params)

    point_clouds, _ = load_dataset("./DataGeneration/id_path_data.yaml", "./DataGeneration/id_pc_data.yaml")
    batch_size = conf.CAEParams.batch_size

    print("start training")
    # training
    for epoch in tqdm(range(conf.CAEParams.epoch_num)):
        losses = []
        for i in range(0, len(point_clouds), batch_size):
            cae.decoder.zero_grad()
            cae.encoder.zero_grad()
            input_clouds_pre: np.array = point_clouds[i:min(i + batch_size, len(point_clouds) - 1)]
            input_clouds: torch.Tensor = torch.from_numpy(input_clouds_pre).to(torch.float32).to(dev)

            latents = cae.encode(input_clouds)
            outputs = cae.decoder(latents)
            loss = cae.loss_function(xs=input_clouds, xs_hat=outputs)

            loss.backward()
            optim.step()
            losses.append(loss.detach().item())
        loss_mean = sum(losses) / len(losses)
        writer.add_scalar("loss_mean", loss_mean, epoch)
    print("avg loss (last step): {}".format(loss_mean))
    writer.close()

    # save models
    torch.save(cae.encoder.cpu().state_dict(), "models/tmp/encoder.pth")
    torch.save(cae.decoder.cpu().state_dict(), "models/tmp/decoder.pth")

    # evaluation
    for point_cloud in point_clouds[0:8]:
        input_clouds: torch.Tensor = torch.from_numpy(point_cloud).to(torch.float32).to(dev)
        latents = cae.encode(input_clouds)
        outputs = cae.decoder(latents)

        pc_hat = outputs.detach().numpy().reshape(conf.TrainingDataParams.point_cloud_num, 2)
        pc = point_cloud.reshape(conf.TrainingDataParams.point_cloud_num, 2)

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.scatter(pc[:, 0], pc[:, 1], label="pc")
        ax2.scatter(pc_hat[:, 0], pc_hat[:, 1], label="pc_hat", color="red")
        fig.legend()
        plt.show()


def main():
    train()


if __name__ == '__main__':
    main()
