import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    weights = ['(0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5)',
               '(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5)',
               '(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)']
    weights_label = ['(1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5)',
               '(1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0)',
               '(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)']
    n_epochs = 6
    figuresize = (8, 8)
    font = 20

    all_data = []
    for w in weights:
        data = pd.read_csv(f'saved_models/results_2024-10-24_n_60_l_5e-05_lp_False_w_{w}_b_1.csv')
        all_data.append(data)
    
    plt.figure(figsize=(12, 8))
    plt.xlabel('Epoch', fontsize=font)
    plt.ylabel('Loss', fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    for i, d in enumerate(all_data):
        plt.errorbar(range(1, n_epochs + 1), d['mean_train_loss'], yerr=d['std_train_loss'], marker='o',
                    markersize=8, capsize=5, label=weights_label[i])
    plt.legend(fontsize=font)
    plt.show()

    TITLE = ['Training', 'ID', 'OOD 1', 'OOD 2']
    DATA =  ['train', 'test', 'ood_1', 'ood_2']
    METRIC = ['precision', 'recall', 'f1']
    YLABEL = [r'$P$', r'$R$', r'$F_1$']

    for J, T in enumerate(TITLE):
        for I, Y in enumerate(YLABEL):
            plt.figure(figsize=figuresize)
            plt.subplots_adjust(left=0.2)
            plt.title(T, fontsize=font)
            plt.xlabel('Epoch', fontsize=font)
            plt.ylabel(Y, fontsize=font)
            plt.xticks(fontsize=font)
            plt.yticks(fontsize=font)
            for i, d in enumerate(all_data):
                plt.errorbar(range(1, n_epochs + 1), d[f'mean_{DATA[J]}_{METRIC[I]}'], yerr=d[f'std_{DATA[J]}_{METRIC[I]}'], marker='o',
                            markersize=8, capsize=5, label=weights_label[i])
            # plt.legend(fontsize=font)
            plt.show()


if __name__ == "__main__":
    main()