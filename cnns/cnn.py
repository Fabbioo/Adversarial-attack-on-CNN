from abc import ABC, abstractmethod

class CNN(ABC):
    
    @abstractmethod
    def get_model(self) -> any:
        pass
    
    @abstractmethod
    def do_inference(self) -> any:
        pass