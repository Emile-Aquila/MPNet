from torch.autograd import Variable
import torch
import torch.nn as nn
from models import EncoderNetwork, DecoderNetwork


class CAE:
    mse: nn.MSELoss

    def __init__(self):
        self.mse = nn.MSELoss(size_average=True)
        self.encoder = EncoderNetwork

    def loss_function(self, xs, xs_hat, model_weights, lamd):
        loss = self.mse(xs, xs_hat)
        reconstruction_loss = torch.sum(Variable(model_weights)**2, dim=1).sum()  # TODO : 要確認
        return loss + reconstruction_loss * lamd

    def




def load_dataset(N, NP):
    pass
