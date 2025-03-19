import os
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from torch.nn import functional as F
import numpy as np


def planar_metrics(logits, labels, num_classes):       # logits:[batch_size, num_classes]   labels:[batch_size, ]
    # accuracy
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
    balanced_acc = balanced_accuracy_score(labels.numpy(), predicted_classes.numpy())

    # macro-average area under the cureve (AUC) scores
    probs = F.softmax(logits, dim=1)
    if num_classes > 2:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
    else:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())

    # weighted f1-score
    f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')

    # quadratic weighted Kappa
    kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')

    # macro specificity 
    specificity_list = []
    for class_idx in range(num_classes):
        true_positive = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() == class_idx))
        true_negative = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() != class_idx))
        false_positive = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() == class_idx))
        false_negative = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() != class_idx))

        specificity = true_negative / (true_negative + false_positive)
        specificity_list.append(specificity)

    macro_specificity = np.mean(specificity_list)

    # confusion matrix
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())

    return {'accuracy': accuracy, 
            'bal_accuracy': balanced_acc, 
            'auc': auc, 
            'f1': f1, 
            'kappa': kappa, 
            'macro_specificity': macro_specificity, 
            'confusion_mat': confusion_mat}