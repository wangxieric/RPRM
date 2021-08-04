from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from operator import itemgetter
import torch

def average_precision_k(targets, predictions, k):
    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def evaluate(ground_truth, predict_results, ks):
    # precision and recall shape batch_size * k
    precision = torch.from_numpy(np.array([0.0] * len(ks)))
    recall = torch.from_numpy(np.array([0.0] * len(ks)))
    precision_u = [[],[],[]]
    recall_u = [[],[],[]]
    sorted_meta_prediction = torch.argsort(predict_results, dim=0, descending=True)
    sorted_meta_prediction = sorted_meta_prediction.squeeze()
    for i, k in enumerate(ks):
        pred = sorted_meta_prediction[:k].tolist()
        num_hit = len(set(pred).intersection(set(ground_truth)))
        precision[i] += float(num_hit) / len(pred)
        precision_u[i].append(float(num_hit) / len(pred))
        recall[i] += float(num_hit) / len(ground_truth)
        recall_u[i].append(float(num_hit) / len(ground_truth))
    
    k = 10
    n_pred = 10
    n_lab = len(ground_truth)
    n = min(max(n_pred, n_lab),k)
    arange = np.arange(n, dtype=np.float32)
    arange = arange[:n_pred]
    denom = np.log2(arange + 2.)
    gains = 1. / denom
    sorted_meta_prediction = sorted_meta_prediction.detach().clone().cpu().numpy()
    dcg_mask = np.in1d(sorted_meta_prediction[:n], ground_truth)
    dcg = gains[dcg_mask].sum()
    
    max_dcg = gains[arange < n_lab].sum()
    ndcg = dcg / max_dcg 
    
    res = sorted_meta_prediction[:10]
    apk = average_precision_k(ground_truth, res, k=np.inf)
    return precision, recall, apk, precision_u, recall_u, ndcg

