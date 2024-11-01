import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    csvs2plot = ['saved_models/results_2024-09-24_n_60_l_5e-05_lp_False_w_(0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5)_b_1.csv',
                'saved_models/results_2024-09-24_n_60_l_0.001_lp_True_w_(0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5)_b_1.csv']
    labels = ['Fine-tuning', 'Linear probing']
    # colors = ['red', 'blue']
    figuresize = (8, 8)
    font = 20

    all_data = []
    for csv in csvs2plot:
        data = pd.read_csv(csv)
        all_data.append(data)
    n_epochs = [len(all_data[0]), len(all_data[1])]
    plt.figure(figsize=(12, 8))
    # plt.title('Training loss', fontsize=font)
    plt.xlabel('Epoch', fontsize=font)
    plt.ylabel('Loss', fontsize=font)
    plt.xticks(np.arange(0, 21, step=5), fontsize=font)
    plt.yticks(fontsize=font)
    for i, d in enumerate(all_data):
        plt.errorbar(range(1, n_epochs[i] + 1), d['mean_train_loss'], yerr=d['std_train_loss'], marker='o',
                    markersize=8, capsize=5, label=labels[i])
    plt.legend(fontsize=font)
    plt.show()

    # Fine-tuning, precision
    i = 0
    d = all_data[0]
    plt.figure(figsize=figuresize)
    plt.title(labels[i], fontsize=font)
    plt.xlabel('Epoch', fontsize=font)
    plt.ylabel(r'$P$', fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_train_precision'], yerr=d['std_train_precision'], marker='o',
                markersize=8, capsize=5, label='Training')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_test_precision'], yerr=d['std_test_precision'], marker='o',
                markersize=8, capsize=5, label='ID')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_1_precision'], yerr=d['std_ood_1_precision'], marker='o',
                markersize=8, capsize=5, label='OOD 1')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_2_precision'], yerr=d['std_ood_2_precision'], marker='o',
                markersize=8, capsize=5, label='OOD 2')
    plt.legend(fontsize=font)
    plt.show()

    # Fine-tuning, recall
    plt.figure(figsize=figuresize)
    plt.title(labels[i], fontsize=font)
    plt.xlabel('Epoch', fontsize=font)
    plt.ylabel(r'$R$', fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_train_recall'], yerr=d['std_train_recall'], marker='o',
                markersize=8, capsize=5, label='Training')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_test_recall'], yerr=d['std_test_recall'], marker='o',
                markersize=8, capsize=5, label='ID')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_1_recall'], yerr=d['std_ood_1_recall'], marker='o',
                markersize=8, capsize=5, label='OOD 1')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_2_recall'], yerr=d['std_ood_2_recall'], marker='o',
                markersize=8, capsize=5, label='OOD 2')
    plt.legend(fontsize=font)
    plt.show()
    
    # Fine-tuning, F1
    plt.figure(figsize=figuresize)
    plt.title(labels[i], fontsize=font)
    plt.xlabel('Epoch', fontsize=font)
    plt.ylabel(r'$F_1$', fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_train_f1'], yerr=d['std_train_f1'], marker='o',
                markersize=8, capsize=5, label='Training')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_test_f1'], yerr=d['std_test_f1'], marker='o',
                markersize=8, capsize=5, label='ID')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_1_f1'], yerr=d['std_ood_1_f1'], marker='o',
                markersize=8, capsize=5, label='OOD 1')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_2_f1'], yerr=d['std_ood_2_f1'], marker='o',
                markersize=8, capsize=5, label='OOD 2')
    plt.legend(fontsize=font)
    plt.show()

    # Linear probing, precision
    i = 1
    d = all_data[1]
    plt.figure(figsize=figuresize)
    plt.title(labels[i], fontsize=font)
    plt.xlabel('Epoch', fontsize=font)
    plt.ylabel(r'$P$', fontsize=font)
    plt.xticks(np.arange(0, 21, step=5), fontsize=font)
    plt.yticks(fontsize=font)
    
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_train_precision'], yerr=d['std_train_precision'], marker='o',
                markersize=8, capsize=5, label='Training')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_test_precision'], yerr=d['std_test_precision'], marker='o',
                markersize=8, capsize=5, label='ID')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_1_precision'], yerr=d['std_ood_1_precision'], marker='o',
                markersize=8, capsize=5, label='OOD 1')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_2_precision'], yerr=d['std_ood_2_precision'], marker='o',
                markersize=8, capsize=5, label='OOD 2')
    plt.legend(fontsize=font)
    plt.show()

    # Linear probing, recall
    plt.figure(figsize=figuresize)
    plt.title(labels[i], fontsize=font)
    plt.xlabel('Epoch', fontsize=font)
    plt.ylabel(r'$R$', fontsize=font)
    plt.xticks(np.arange(0, 21, step=5), fontsize=font)
    plt.yticks(fontsize=font)
    
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_train_recall'], yerr=d['std_train_recall'], marker='o',
                markersize=8, capsize=5, label='Training')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_test_recall'], yerr=d['std_test_recall'], marker='o',
                markersize=8, capsize=5, label='ID')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_1_recall'], yerr=d['std_ood_1_recall'], marker='o',
                markersize=8, capsize=5, label='OOD 1')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_2_recall'], yerr=d['std_ood_2_recall'], marker='o',
                markersize=8, capsize=5, label='OOD 2')
    plt.legend(fontsize=font)
    plt.show()

    # Linear probing, f1
    plt.figure(figsize=figuresize)
    plt.title(labels[i], fontsize=font)
    plt.xlabel('Epoch', fontsize=font)
    plt.ylabel(r'$F_1$', fontsize=font)
    plt.xticks(np.arange(0, 21, step=5), fontsize=font)
    plt.yticks(fontsize=font)
    
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_train_f1'], yerr=d['std_train_f1'], marker='o',
                markersize=8, capsize=5, label='Training')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_test_f1'], yerr=d['std_test_f1'], marker='o',
                markersize=8, capsize=5, label='ID')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_1_f1'], yerr=d['std_ood_1_f1'], marker='o',
                markersize=8, capsize=5, label='OOD 1')
    plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_2_f1'], yerr=d['std_ood_2_f1'], marker='o',
                markersize=8, capsize=5, label='OOD 2')
    plt.legend(fontsize=font)
    plt.show()

    # plt.figure(figsize=figuresize)
    # plt.xlabel('Epoch', fontsize=font)
    # plt.ylabel('Recall', fontsize=font)
    # plt.xticks(fontsize=font)
    # plt.yticks(fontsize=font)
    # for i, d in enumerate(all_data):
    #     plt.errorbar(range(1, n_epochs[i] + 1), d['mean_train_recall'], yerr=d['std_train_recall'], marker='o',
    #                 markersize=8, capsize=5, color = colors[i], label=f'{labels[i]}, training')
    #     plt.errorbar(range(1, n_epochs[i] + 1), d['mean_test_recall'], yerr=d['std_test_recall'], marker='^',
    #                 markersize=8, capsize=5, color = colors[i], linestyle='dashed', label=f'{labels[i]}, test')
    #     plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_1_recall'], yerr=d['std_ood_1_recall'], marker='s',
    #                 markersize=8, capsize=5, color = colors[i], linestyle='dotted', label=f'{labels[i]}, OOD')
    # plt.legend(fontsize=font)
    # plt.show()

    # plt.figure(figsize=figuresize)
    # plt.xlabel('Epoch', fontsize=font)
    # plt.ylabel('F1-score', fontsize=font)
    # plt.xticks(fontsize=font)
    # plt.yticks(fontsize=font)
    # for i, d in enumerate(all_data):
    #     plt.errorbar(range(1, n_epochs[i] + 1), d['mean_train_f1'], yerr=d['std_train_f1'], marker='o',
    #                 markersize=8, capsize=5, color = colors[i], label=f'{labels[i]}, training')
    #     plt.errorbar(range(1, n_epochs[i] + 1), d['mean_test_f1'], yerr=d['std_test_f1'], marker='^',
    #                 markersize=8, capsize=5, color = colors[i], linestyle='dashed', label=f'{labels[i]}, test')
    #     plt.errorbar(range(1, n_epochs[i] + 1), d['mean_ood_f1'], yerr=d['std_ood_f1'], marker='s',
    #                 markersize=8, capsize=5, color = colors[i], linestyle='dotted', label=f'{labels[i]}, OOD')
    # plt.legend(fontsize=font)
    # plt.show()




if __name__ == "__main__":
    main()