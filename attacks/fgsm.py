from torch import Tensor
from attacks.attack import Attack
from cnns.resnet50 import ResNet50
from losses.crossentropy import CrossEntropyLoss
from globals.GlobalSettings import device

class FGSM(Attack):
    
    def do_attack(self, image: Tensor, model_function: ResNet50, loss_function: CrossEntropyLoss, epsilon: float) -> Tensor:
        
        if epsilon == 0:
            return image
        
        original_image: Tensor = image.clone()

        original_image.requires_grad = True

        output: tuple[int, str, float] = model_function.do_inference(original_image)
        class_id: Tensor = Tensor([output[0]]).long().to(device)

        output: Tensor = model_function.get_model()(original_image)
        model_function.get_model().zero_grad()

        loss: Tensor = loss_function.get_loss_function()(output, class_id)
        loss.backward(retain_graph = True)

        perturbed_image: Tensor = original_image + epsilon * original_image.grad.sign()

        return perturbed_image