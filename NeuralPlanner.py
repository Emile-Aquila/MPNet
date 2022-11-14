import torch
from models import PlannerNetwork
from typing import TypeVar, Generic


State = TypeVar("State")


class NeuralPlanner(Generic[State]):
    Pnet: PlannerNetwork
    device: torch.device

    def __init__(self, PNet: PlannerNetwork, device: torch.device):
        self.device = device
        self.Pnet = PNet.to(self.device)

    def planning(self, start_state: State, goal_state: State, Z, iter_times: int) -> list[State]:
        tau_a, tau_b = list[State]([start_state]), list[State]([goal_state])
        for i in range(iter_times):
            if i % 2 == 0:
                state_new = self.Pnet(tau_a[-1], tau_b[-1], Z)
                tau_a.append(state_new)
            else:
                state_new = self.Pnet(tau_b[-1], tau_a[-1], Z)
                tau_b.append(state_new)
            if self.SteerTo(tau_a[-1], tau_b[-1]):
                return tau_a + tau_b
        print("[WARN] Cannot generating trajectory (NeuralPlanner)")
        return list[State]()

    def SteerTo(self, state1: State, state2: State) -> bool:
        # check connectivity of (state1, state2)
        # TODO : 実装
        return True


