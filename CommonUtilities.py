from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import torchvision
import torchvision.transforms.functional
from torch import Tensor

class CommonUtilities:

    @staticmethod
    def preprocess(image: Tensor) -> Tensor:
        image = torch.clamp(image, 0, 255)
        image = image.float() / 255
        preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ])
        image = preprocessing(image)
        image = image.unsqueeze(0)
        return image

    @staticmethod
    def postprocess(image: Tensor) -> Tensor:
        postprocessing = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std = [1/0.229, 1/0.224, 1/0.225]
            )
        ])
        image = postprocessing(image)
        image = torch.clamp(image, 0, 1) * 255
        return image
    
    @staticmethod
    def str2tensor(image: str) -> Tensor:
        return torchvision.transforms.ToTensor()(Image.open(image))
    
    @staticmethod
    def tensor2array(tensor: Tensor) -> np.ndarray:
        return (tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    
    @staticmethod
    def visualize(image: str | Tensor | np.ndarray) -> None:
        if isinstance(image, str):
            image_to_display: np.ndarray = mpimg.imread(image)
        elif isinstance(image, Tensor):
            if image.dim() == 4:
                image_to_display: np.ndarray = CommonUtilities.tensor2array(image.squeeze(0))
            elif image.dim() == 3:
                image_to_display: np.ndarray = CommonUtilities.tensor2array(image)
            else:
                raise Exception(f"Tensore con una errata dimensionalità: {image.size()}")
        elif isinstance(image, np.ndarray):
            pass # type(image) = np.ndarray
        else:
            raise Exception(f"Immagine nel formato errato: {type(image)}")
        plt.figure(figsize = (8, 8))
        plt.axis("off")
        plt.imshow(image_to_display)