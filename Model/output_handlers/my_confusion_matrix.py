# IMPORT

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

########################################################################################################################

def confusion_matrix_print(val_predictions, emotion_labels): #, run_path, run_counter):

    real_val_data, labels_prediction = zip(*val_predictions)

    real_val_data = [x for tupla in real_val_data for x in tupla]
    labels_prediction = [x for tupla in labels_prediction for x in tupla]

    confusion_mat = confusion_matrix(real_val_data, labels_prediction, labels=range(len(emotion_labels)))

    confusion_mat_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)

    confusion_mat_display.plot()
    plt.savefig("confusionVERA")
    # plt.close()