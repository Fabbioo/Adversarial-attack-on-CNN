import torch
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

device: str = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "cpu" # TODO sistema per capire se mettere su cpu o mps

x_figure_plot_size: int = 16
y_figure_plot_size: int = 8
fig_size: tuple[int, int] = (x_figure_plot_size, y_figure_plot_size)

plt.rcParams.update({
    "figure.figsize": fig_size,         # Dimensione della figura.
    "figure.autolayout": True,          # Regolazione automatica delle dimensioni della figura.
    "figure.titlesize": 20,             # Dimensione del titolo associato ad ogni figura (plt.suptitle()).
    "axes.titlesize": 20,               # Dimensione del titolo associato ad ogni grafico all'interno di una figura (plt.title()).
    "axes.labelsize": 20,               # Dimensione delle etichette sia sull'asse x sia sull'asse y.
    "xtick.labelsize": 15,              # Dimensione dei riferimenti sull'asse x.
    "ytick.labelsize": 15,              # Dimensione dei riferimenti sull'asse y.
    "legend.fontsize": 20,              # Dimensione dei caratteri della legenda.
    "font.family": "times new roman",   # Font utilizzata per i testi.
})