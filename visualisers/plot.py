from abc import ABC, abstractmethod
from cnns.resnet50 import ResNet50

class Plot(ABC):
    
    @abstractmethod
    def plot_predictions(self, model_function: ResNet50, tripla: tuple, epsilon: float, show_noise: bool = False) -> None:
        pass