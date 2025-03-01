from visualisers.plot import Plot
from CommonUtilities import CommonUtilities
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

class PlotPredictions(Plot):

    def plot_predictions(self, model_function, tripla, show_noise = False):

        outputs_orig: tuple[int, str, float] = model_function.do_inference(tripla[0])
        outputs_pert: tuple[int, str, float] = model_function.do_inference(tripla[2])

        color: str = "green" if outputs_orig[1] == outputs_pert[1] else "red" # Se le due predizioni coincidono stampo una scritta verde, altrimenti rossa
        
        if show_noise:
            images: list = [CommonUtilities.tensor2array(tripla[0]), CommonUtilities.tensor2array(tripla[1]), CommonUtilities.tensor2array(tripla[2])]
            objects: list[str] = ["ORIGINAL", "NOISE", "PERTURBED"]
            plt.figure(figsize = (15, 5))
            gs = gridspec.GridSpec(1, 5, width_ratios = [5, 0.1, 5, 0.1, 5])
            for i in range(len(images) + 2): # + 2 perchè voglio rappresentare sia il + sia il =
                plt.subplot(gs[i])
                match i:
                    case 0:
                        plt.imshow(images[0])
                        plt.title(objects[0] + "\n\n" + str(outputs_orig[0]) + ": " + outputs_orig[1] + f"$\\Rightarrow$ {outputs_orig[2] * 100:.3}%", color = "green")
                    case 1:
                        plt.text(0.5, 0.5, "+", fontsize = 40, ha = "center", fontweight = "bold")
                    case 2:
                        plt.imshow(images[1])
                        plt.title(objects[1])
                    case 3:
                        plt.text(0.5, 0.5, "=", fontsize = 40, ha = "center", fontweight = "bold")
                    case 4:
                        plt.imshow(images[2])
                        plt.title(objects[2] + "\n\n" + str(outputs_pert[0]) + ": " + outputs_pert[1] + f"$\\Rightarrow$ {outputs_pert[2] * 100:.3}%", color = color)
                plt.axis("off")
        else:
            images: list = [CommonUtilities.tensor2array(tripla[0]), CommonUtilities.tensor2array(tripla[2])]
            objects: list[str] = ["ORIGINAL", "PERTURBED"]
            plt.figure()
            for i in range(len(images)):
                plt.subplot(1, len(images), i + 1)
                plt.imshow(images[i])
                match i:
                    case 0:
                        plt.title(objects[0] + "\n\n" + str(outputs_orig[0]) + ": " + outputs_orig[1] + f" → {outputs_orig[2] * 100:.3}%", color = "green")
                    case 1:
                        plt.title(objects[1] + "\n\n" + str(outputs_pert[0]) + ": " + outputs_pert[1] + f" → {outputs_pert[2] * 100:.3}%", color = color)
                plt.axis("off")