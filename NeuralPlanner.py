from typing import TypeVar, Generic
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from CAE import load_dataset, CAE
from models import PlannerNetwork
State = TypeVar("State")


class NeuralPlanner(Generic[State]):
    PNet: PlannerNetwork
    device: torch.device

    def __init__(self, PNet: PlannerNetwork):
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.PNet = PNet.to(self.dev)
        self.MSE = nn.MSELoss()

    def planning(self, start_state: State, goal_state: State, Z, iter_times: int) -> list[State]:
        tau_a, tau_b = list[State]([start_state]), list[State]([goal_state])
        for i in range(iter_times):
            if i % 2 == 0:
                state_new = self.PNet(tau_a[-1], tau_b[-1], Z)
                tau_a.append(state_new)
            else:
                state_new = self.PNet(tau_b[-1], tau_a[-1], Z)
                tau_b.append(state_new)
            if self.SteerTo(tau_a[-1], tau_b[-1]):
                return tau_a + tau_b
        print("[WARN] Cannot generating trajectory (NeuralPlanner)")
        return list[State]()

    def SteerTo(self, state1: State, state2: State) -> bool:
        # check connectivity of (state1, state2)
        # TODO : 実装
        return True


@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
def train(conf: DictConfig):
    # import training data
    point_clouds, path_id_tuples = load_dataset("./DataGeneration/id_path_data.yaml",
                                                "./DataGeneration/id_pc_data.yaml")
    print(point_clouds.shape)

    # define
    cae: CAE = CAE(input_size=conf.TrainingDataParams.point_cloud_num * conf.CAEParams.coordinates_dim,
                   output_size=conf.CAEParams.latent_space_size)
    cae.encoder.load_state_dict(torch.load("./models/encoder.pth"))
    cae.encoder.to(cae.dev)
    cae.decoder.load_state_dict(torch.load("./models/decoder.pth"))
    cae.decoder.to(cae.dev)

    p_net = PlannerNetwork(input_size=conf.CAEParams.latent_space_size + 2 * conf.PNetParams.coordinates_dim,
                           output_size=conf.PNetParams.coordinates_dim)
    neural_planner: NeuralPlanner = NeuralPlanner(PNet=p_net)
    optimizer = torch.optim.Adagrad(neural_planner.PNet.parameters())
    print("Planner Network: {}".format(neural_planner.PNet.models))

    writer = SummaryWriter(log_dir="./logs/PNet")

    # 事前処理
    latent_spaces: list[torch.Tensor] = []
    for point_cloud in point_clouds:
        with torch.no_grad():
            latent_space: torch.Tensor = cae.encode(torch.Tensor(point_cloud).to(cae.dev))
        latent_spaces.append(latent_space)

    # training
    batch_size = conf.PNetParams.batch_size
    mse = nn.MSELoss(size_average=True)

    for epoch in tqdm(range(conf.PNetParams.epoch_num)):
        epoch_losses = []
        for i in range(0, len(path_id_tuples), batch_size):
            neural_planner.PNet.zero_grad()
            path_id_tuples_pre: np.array = path_id_tuples[i:min(i + batch_size, len(path_id_tuples) - 1)]
            traj_losses = []
            for path_id_tuple in path_id_tuples_pre:
                traj, field_id = path_id_tuple  # traj: np.array
                traj = torch.Tensor(traj).to(neural_planner.dev)
                x_target = traj[-1]
                x_hats = torch.cat(
                    [neural_planner.PNet.forward(x, x_target, latent_spaces[field_id]) for x in traj[0:-1]], dim=0)
                traj_losses.append(mse(x_hats, traj[1:].flatten()))
            loss = torch.stack(traj_losses).mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        writer.add_scalar("epoch_loss", sum(epoch_losses) / len(epoch_losses), epoch)
    writer.close()
    torch.save(neural_planner.PNet.cpu().state_dict(), "models/tmp/pnet.pth")


def main():
    train()


if __name__ == '__main__':
    main()
