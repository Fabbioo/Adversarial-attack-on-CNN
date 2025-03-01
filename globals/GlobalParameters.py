class GlobalParameters:

    epsilon_fgsm: float = 0.3

    epsilon_ifgsm: float = 0.3
    iters_ifgsm: int = 100
    alpha_ifgsm: float = epsilon_ifgsm/iters_ifgsm 

    epsilon_pgd: float = 0.3
    iters_pgd: int = 100
    alpha_pgd: float = epsilon_pgd/iters_pgd