import torch.nn as nn

from losses.loss import Loss

class CrossEntropyLoss(Loss):

    def __init__(self) -> None:
        self.__loss: any = nn.CrossEntropyLoss()
    
    def get_loss_function(self) -> any:
        return self.__loss