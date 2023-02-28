import numpy as np
import torch
from CAE import CAE
from models import PlannerNetwork
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from DataGeneration.TrajectryGeneration.objects.field import Field, Rectangle, Point2D
from DataGeneration.data_generaton import generate_point_cloud
State = list[float]

def convert_State(state: State) -> Point2D:
    return Point2D(state[0], state[1], 0.0)


class MPNet:
    pnet: PlannerNetwork
    cae: CAE
    device: torch.device
    steer_to_div_num: int
    planning_max_step: int
    field: Field

    def __init__(self, PNet: PlannerNetwork, cae: CAE, planning_max_step: int, steer_to_div_num: int, field: Field):
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pnet = PNet.to(self.dev)
        self.cae = cae
        self.steer_to_div_num = steer_to_div_num
        self.planning_max_step = planning_max_step
        self.field = field

    def _check_collision(self, state: State):
        # 衝突判定。_steer_toで使われる
        if self.field.check_collision(convert_State(state)):
            return True
        return False

    def _lazy_state_contraction(self, path: list):
        # 直接つなげる経路点同士をつないで、無駄な経路点を除外。
        if len(path) <= 2:
            return
        point_last: State = path[-1]
        for i in range(len(path[0:-2])):
            if not self._steer_to(path[i], point_last, self.steer_to_div_num):
                del path[i+1:-1]
                return

    def _is_feasible(self, traj: list) -> bool:
        # 経路が衝突無く連続に繋がっているか判定。鋭角になってたらFalseを返す
        for i in range(1, len(traj)-2):
            v1 = [traj[i-1][j] - traj[i][j] for j in range(len(traj[i]))]
            v2 = [traj[i+1][j] - traj[i][j] for j in range(len(traj[i]))]
            prod = 0.0
            for x1, y1 in zip(v1, v2):
                prod += x1*y1
            if prod > 0.0:
                return False
        return True

    def _steer_to(self, point1: State, point2: State, step_num: int) -> bool:
        # point1, point2をつなぐ直線経路が障害物と干渉してないか判定
        for delta in np.linspace(0.0, 1.0, step_num):
            point = [p1*delta + p2*(1.0-delta) for p1, p2 in zip(point1, point2)]
            if self._check_collision(state=point):
                return False
        return True

    def _replanning(self, traj: list, z: torch.Tensor) -> list:
        # 経路の再生成
        traj_n, traj_ans = [], []
        for i in range(len(traj)-1):
            if not self._check_collision(traj[i]):
                traj_n.append(traj[i])
        traj_ans.append(traj_n[0])

        for i in range(len(traj_n)-1):
            if self._steer_to(traj_n[i], traj_n[i+1], self.steer_to_div_num):
                traj_ans.append(traj[i+1])
            else:  # re-planning
                traj_f, traj_b = [traj_n[i]], [traj_n[i+1]]
                for j in range(50):
                    point_f = torch.Tensor(traj_f[-1]).to(self.dev)
                    point_b = torch.Tensor(traj_b[-1]).to(self.dev)
                    if j % 2 == 0:
                        x_new: torch.Tensor = self.pnet.forward(point_f, point_b, z)
                        traj_f.append(x_new.tolist())
                    else:
                        x_new: torch.Tensor = self.pnet.forward(point_b, point_f, z)
                        traj_b.append(x_new.tolist())
                    if self._steer_to(traj_f[-1], traj_b[-1], self.steer_to_div_num):
                        traj_ans += traj_f[1:] + list(reversed(traj_b))
                    else:
                        return list()
        return traj_ans


    def _neural_planner(self, start_point: State, goal_point: State, z: torch.Tensor, max_step: int) -> list:
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
            if self._steer_to(traj_forward[-1], traj_back[-1], self.steer_to_div_num):  # 接続判定
                return traj_forward + list(reversed(traj_back))
            print(traj_forward, traj_back)
        # return list()
        return traj_forward + list(reversed(traj_back))  # TODO: for Debug


    def planning(self, start_point: State, goal_point: State, point_cloud: np.array) -> list:
        z: torch.Tensor = self.cae.encoder(torch.from_numpy(point_cloud.flatten()).float().to(self.dev))  # latent space
        traj: list = self._neural_planner(start_point, goal_point, z, self.planning_max_step)
        if len(traj) == 0:
            return traj
        else:
            return traj  # TODO: for debug
            self._lazy_state_contraction(traj)
            if self._is_feasible(traj):
                return traj
            else:
                traj_new = self._replanning(traj, z)
                self._lazy_state_contraction(traj_new)
                if self._is_feasible(traj_new):
                    return traj_new
        return list()


def generate_test_field() -> Field:
    field = Field(w=20.0, h=20.0, center=True)
    obstacle_l: float = 3.0
    points = [[0.0, 0.0], [2.0, -2.0], [4.0, -5.0], [-2.0, 3.0], [-5.0, 3.0]]
    for pt in points:
        field.add_obstacle(Rectangle(x=pt[0], y=pt[1], w=obstacle_l, h=obstacle_l, theta=0.0, fill=True))

    return field



@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
def main(conf: DictConfig):
    # define
    cae: CAE = CAE(input_size=conf.TrainingDataParams.point_cloud_num * conf.CAEParams.coordinates_dim,
                   output_size=conf.CAEParams.latent_space_size)
    cae.encoder.load_state_dict(torch.load("./models/encoder.pth"))
    cae.encoder.to(cae.dev)
    cae.decoder.load_state_dict(torch.load("./models/decoder.pth"))
    cae.decoder.to(cae.dev)

    p_net = PlannerNetwork(input_size=conf.CAEParams.latent_space_size + 2 * conf.PNetParams.coordinates_dim,
                           output_size=conf.PNetParams.coordinates_dim)
    p_net.load_state_dict(torch.load("./models/pnet.pth"))
    p_net.to(cae.dev)

    field: Field = generate_test_field()
    field.plot()

    mp_net: MPNet = MPNet(PNet=p_net, cae=cae,
                          planning_max_step=conf.MPNetParams.planning_max_step,
                          steer_to_div_num=conf.MPNetParams.steer_to_div_num,
                          field=field)

    point_cloud: np.array = generate_point_cloud(field, num_pc=conf.TrainingDataParams.point_cloud_num, object_num=5)
    z: torch.Tensor = cae.encoder(torch.from_numpy(point_cloud.flatten()).float().to(cae.dev))  # latent space
    pc_hat = cae.decoder(z)
    pc_hat = pc_hat.detach().numpy().reshape(conf.TrainingDataParams.point_cloud_num, 2)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], label="pc")
    ax2.scatter(pc_hat[:, 0], pc_hat[:, 1], label="pc_hat", color="red")
    plt.show()

    start_point, goal_point = [-3.0, -3.0], [5.0, 5.0]

    traj = mp_net.planning(start_point, goal_point, point_cloud)
    print(traj)
    field.plot_path([Point2D(p[0], p[1], 0.0) for p in traj],
                    Point2D(start_point[0], start_point[1]),
                    Point2D(goal_point[0], goal_point[1]), show=True)

if __name__ == '__main__':
    # main()
    main()