from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights

from cnns.cnn import CNN
from globals.GlobalSettings import device

class ResNet50(CNN):

    def __init__(self) -> None:
        self.__model: any = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2).eval().to(device)
    
    def get_model(self) -> any:
        return self.__model
    
    def do_inference(self, image: Tensor) -> tuple[int, str, float]:
        output: any = self.__model(image).squeeze(0).softmax(0)
        class_id: int = output.argmax().item()
        class_name: str = ResNet50_Weights.IMAGENET1K_V2.meta["categories"][class_id]
        class_conf: float = output[class_id].item()
        return (class_id, class_name, class_conf)