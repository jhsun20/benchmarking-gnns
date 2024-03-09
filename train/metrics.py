import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np

def accuracy_CO(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    return acc/len(scores)

def binary_f1_score(scores, targets, split='train'):
    """Computes the F1 score using scikit-learn for binary class labels. 
    
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    if split == 'test':
        print("y_true:", np.sum(y_true))
        print("y_pred:", np.sum(y_pred))
    return f1_score(y_true, y_pred)