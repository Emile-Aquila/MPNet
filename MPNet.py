import numpy as np
import torch
from CAE import CAE
from models import PlannerNetwork


class MPNet:
    pnet: PlannerNetwork
    cae: CAE
    device: torch.device

    def __init__(self, PNet: PlannerNetwork, cae: CAE):
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pnet = PNet.to(self.dev)
        self.cae = cae

    def _check_collision(self, point1: list[float], point2: list[float]):
        # TODO: 衝突判定どうするか？
        pass

    def _lazy_state_contraction(self, path: list):
        # 直接つなげる経路点同士をつないで、無駄な経路点を除外。
        if len(path) <= 2:
            return
        point_last: list[float] = path[-1]
        for i in range(path[0:-2]):
            if not self._check_collision(path[i], point_last):
                del path[i+1:-1]
                return

    def _is_feasible(self, traj: list):
        # 経路が衝突無く連続に繋がっているか判定
        pass

    def _steer_to(self, point1: list[float], point2: list[float]):
        # point1, point2をつなぐ直線経路が障害物と干渉してないか判定
        pass

    def _replanning(self, traj: list, z: torch.Tensor) -> list:
        # 経路の再生成
        pass

    def _neural_planner(self, start_point: list[float], goal_point: list[float], z: torch.Tensor, max_step: int) -> list:
        traj_forward, traj_back = [start_point], [goal_point]
        for i in range(max_step):
            point_forward = torch.Tensor(traj_forward[-1]).to(self.dev)
            point_back = torch.Tensor(traj_back[-1]).to(self.dev)
            if i % 2 == 0:
                x_new: torch.Tensor = self.pnet.forward(point_forward, point_back, z)
                traj_forward.append(x_new.tolist())
            else:
                x_new: torch.Tensor = self.pnet.forward(point_back, point_forward, z)
                traj_back.append(x_new.tolist())
            if self._steer_to(traj_forward[-1], traj_back[-1]):  # 接続判定
                return traj_forward + traj_back
        return list()


    def planning(self, start_point: list[float], goal_point: list[float], point_cloud: np.array, max_step: int) -> list:
        z: torch.Tensor = self.cae.encoder(torch.from_numpy(point_cloud).to(self.dev))  # latent space
        traj: list = self._neural_planner(start_point, goal_point, z, max_step)
        if len(traj) == 0:
            return traj
        else:
            self._lazy_state_contraction(traj)
            if self._is_feasible(traj):
                return traj
            else:
                traj_new = self._replanning(traj, z)
                self._lazy_state_contraction(traj_new)
                if self._is_feasible(traj_new):
                    return traj_new
        return list()



def main():
    pass

if __name__ == '__main__':
    main()
