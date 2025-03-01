from abc import ABC, abstractmethod

class Loss(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def get_loss_function(self) -> any:
        pass