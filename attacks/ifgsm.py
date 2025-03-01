import torch

from torch import Tensor
from attacks.attack import Attack
from cnns.resnet50 import ResNet50
from losses.crossentropy import CrossEntropyLoss
from globals.GlobalSettings import device

class IFGSM(Attack):

    def do_attack(self, image: Tensor, model_function: ResNet50, loss_function: CrossEntropyLoss, epsilon: float, alpha: float, iter: int) -> Tensor:

        if epsilon == 0:
            return image
            
        original_image: Tensor = image.clone()
        perturbed_image: Tensor = image.clone()

        output: tuple[int, str, float] = model_function.do_inference(original_image)
        class_id = Tensor([output[0]]).long().to(device)    
                
        for _ in range(iter):

            perturbed_image.requires_grad = True
                
            output: Tensor = model_function.get_model()(perturbed_image)
            model_function.get_model().zero_grad()
                
            loss: Tensor = loss_function.get_loss_function()(output, class_id)
            loss.backward()
                
            perturbed_image: Tensor = perturbed_image + alpha * perturbed_image.grad.sign()
            perturbed_image = torch.clamp(perturbed_image, perturbed_image - epsilon, perturbed_image + epsilon)

            perturbed_image = perturbed_image.detach()

        return perturbed_image