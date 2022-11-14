import torch
from models import PlannerNetwork, EncoderNetwork
from NeuralPlanner import NeuralPlanner
from typing import TypeVar, Generic

State = TypeVar("State")


class MPNet(Generic[State]):
    device: torch.device
    planner: NeuralPlanner[State]

    def __init__(self, input_size: int, output_size: int):
        self.device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
        self.PNet = PlannerNetwork(input_size, output_size).to(self.device)
        self.ENet = EncoderNetwork(input_size, output_size).to(self.device)
        self.planner = NeuralPlanner(PNet=self.PNet, device=self.device)

        print("=====INFORMATION=====")
        print("torch version: {}".format(torch.__version__))
        print("CUDA is available: {}".format(torch.cuda.is_available()))
        if torch.cuda.is_available():
            print("GPU: {}".format(torch.cuda.get_device_name()))

    def planning(self, start_state: State, goal_state: State, obstacles) -> list[State] | None:
        z = self.ENet(obstacles)
        traj = self.planner.planning(start_state, goal_state, z, iter_times=1000)
        if len(traj) == 0:
            print("[ERROR] Cannot generate trajectory (MPNet)")
            return None
        if self.isFeasible(traj):
            return traj
        traj_pre = self.Replanning(traj, z)
        traj = self.LazyStatesContraction(traj_pre)
        if self.isFeasible(traj):
            return traj
        else:
            print("[ERROR] Cannot generate trajectory (MPNet)")
            return None

    def LazyStatesContraction(self, traj: list[State]) -> list[State]:
        # 直接繋げる経路点を接続して, 無駄な経路点を除外
        # TODO : 実装
        return traj

    def Replanning(self, traj: list[State], Z) -> list[State]:
        #
        # TODO : 実装
        return traj

    def isFeasible(self, traj: list[State]) -> bool:
        # まともな経路かチェック
        # TODO : 実装
        return True
