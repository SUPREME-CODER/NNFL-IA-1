import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics     import roc_auc_score
sns.set(style = "white", palette = "deep")

def countplots(df, columns, titles, labels, ylabel, dim, row, col, orient = "h", hue = None):
    fig = plt.figure(figsize = dim, facecolor = "white")
    count = 0
    for c, column in enumerate(columns):
        count += 1
        ax = fig.add_subplot(row, col, count)
        title = titles[c]
        plt.title(f"{title}", size = 18)
        sns.countplot(x = column, data = df, orient = orient, hue = hue)
        plt.xlabel(f"{labels[c]}", size = 16)
        plt.ylabel(f"{ylabel}", size = 16)
        plt.xticks(size = 14)
        plt.yticks(size = 14)
    plt.tight_layout()
    plt.show();

def roc_curve(model_prob, X_test, y_test, y_predicted, title, dim, roc_color = "darkorange", baseline_color = "darkblue"):
    model_prob = [i[0] for i in model_prob.predict_proba(X_test)]
    model_pred_df = pd.DataFrame({"true_values": y_test, "pred_probs": model_prob})
    thresholds = np.linspace(0, 1, 500) 
    def true_positive_rate(df, true_col, pred_prob_col, threshold):
        true_positive = df[(df[true_col] == 1) & (df[pred_prob_col] >= threshold)].shape[0]
        false_negative = df[(df[true_col] == 1) & (df[pred_prob_col] < threshold)].shape[0]
        return true_positive / (true_positive + false_negative)
    def false_positive_rate(df, true_col, pred_prob_col, threshold):
        true_negative = df[(df[true_col] == 0) & (df[pred_prob_col] <= threshold)].shape[0]
        false_positive = df[(df[true_col] == 0) & (df[pred_prob_col] > threshold)].shape[0]
        return 1 - (true_negative / (true_negative + false_positive))
    tpr_values = [true_positive_rate(model_pred_df, "true_values", "pred_probs", prob) for prob in thresholds]
    fpr_values = [false_positive_rate(model_pred_df, "true_values", "pred_probs", prob) for prob in thresholds]
    plt.figure(figsize = dim, facecolor = "white")
    plt.plot(fpr_values, tpr_values, color = roc_color, label = "ROC Curve")
    plt.plot(np.linspace(0, 1, 500), np.linspace(0, 1, 500), color = baseline_color, label = "Baseline")
    rocauc_score = round(roc_auc_score(y_test, y_predicted), 5)
    plt.title(f"{title} With A Score of {rocauc_score}", fontsize = 18)
    plt.ylabel("Sensitivity", size = 16)
    plt.xlabel("1 - Specificity", size = 16)
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    plt.legend(bbox_to_anchor = (1.04, 1), loc = "upper left", fontsize = 16)
    plt.tight_layout()