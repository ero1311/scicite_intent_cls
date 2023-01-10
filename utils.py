import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def create_conf_matrix_fig(y_true, y_pred):
    classes = ["method", "background", "result"]
    cf_mat = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_mat, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    return sns.heatmap(df_cm, annot=True).get_figure()