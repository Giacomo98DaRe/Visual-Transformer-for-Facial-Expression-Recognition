# IMPORT

import os
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

########################################################################################################################

def loss_history_graph(loss_train, loss_val, run_path, run_counter):
    figure = plt.figure(figsize=(15, 5))
    plt.plot(loss_train, label='Training', alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(loss_val, label='Validation', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Loss function')  # Titolo cambiabile in base alla loss function che si sceglie
    plt.grid(alpha=.3)

    figure.savefig(os.path.join(run_path, f"loss_story_{run_counter}"))
    plt.close()

def acc_history_graph(acc_train, acc_val, run_path, run_counter):
    figure = plt.figure(figsize=(15, 5))
    plt.plot(acc_train, label='Training', alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(acc_val, label='Validation', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    figure.savefig(os.path.join(run_path, f"acc_story_{run_counter}"))
    plt.close()

