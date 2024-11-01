import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    figuresize = (8, 8)
    font = 20
    csv2plot = ['saved_models/results_2024-10-21_n_60_l_5e-05_lp_False_w_(0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5)_b_1.csv',
    'saved_models/2024-10-21_n_60_e_20_l_5e-05_lp_False_w_(0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5)_b_8_s_111/results.csv']
    all_data = []
    for c in csv2plot:
        data = pd.read_csv(c)
        all_data.append(data)
    
    plt.figure(figsize=(12, 8))
    plt.xlabel('Epoch', fontsize=font)
    plt.ylabel('Loss', fontsize=font)
    plt.xticks(np.arange(0, 21, step=5), fontsize=font)
    plt.yticks(fontsize=font)
    
    plt.errorbar(range(1, 6 + 1), all_data[0]['mean_train_loss'], yerr=all_data[0]['std_train_loss'], marker='o',
                markersize=8, capsize=5, label=rf'$B = 1$')
    plt.plot(range(1, 20 + 1), all_data[1]['train_loss'], 'o-', label=rf'$B = 8$', markersize=8)
    plt.legend(fontsize=font)
    plt.show()

    TITLE = ['Training', 'ID', 'OOD 1', 'OOD 2']
    DATA =  ['train', 'test', 'ood_1', 'ood_2']
    METRIC = ['precision', 'recall', 'f1']
    YLABEL = [r'$P$', r'$R$', r'$F_1$']

    for J, T in enumerate(TITLE):
        for I, Y in enumerate(YLABEL):
            plt.figure(figsize=figuresize)
            plt.title(T, fontsize=font)
            plt.xlabel('Epoch', fontsize=font)
            plt.ylabel(Y, fontsize=font)
            plt.xticks(np.arange(0, 21, step=5), fontsize=font)
            plt.yticks(fontsize=font)
            plt.errorbar(range(1, 6 + 1), all_data[0][f'mean_{DATA[J]}_{METRIC[I]}'], yerr=all_data[0][f'std_{DATA[J]}_{METRIC[I]}'], marker='o',
                        markersize=8, capsize=5, label=rf'$B = 1$')
            plt.plot(range(1, 20 + 1), all_data[1][f'{DATA[J]}_{METRIC[I]}'], 'o-', label=rf'$B = 8$', markersize=8)
            plt.legend(fontsize=font)
            plt.show()


if __name__ == "__main__":
    main()