import torch


def bpr_loss(pos_pred, neg_pred, reg = 0.0, mask=None):
    """
    Bayesian Personalised Ranking (A pairwise ranking loss function)
    :param pos_pred:  predicted scores for known positive items
    :param neg_pred: predicted scores for negatively sampled items
    :param mask: zero the loss of unnecessary comparison
    :return: loss: the mean value of the summed loss
    """
    # log_prob = (torch.log(torch.sigmoid(y_ui - y_uj))).mean()
    sig = torch.sigmoid(pos_pred - neg_pred) + 1e-7
    log_prob = (torch.log(sig)).mean()
    loss = -log_prob
    return loss


def bpr_sep_loss(pos_pred, neg_pred, reg = 0.0, mask=None):
    """
    Bayesian Personalised Ranking (A pairwise ranking loss function)
    :param pos_pred:  predicted scores for known positive items
    :param neg_pred: predicted scores for negatively sampled items
    :param mask: zero the loss of unnecessary comparison
    :return: loss: the mean value of the summed loss
    """
    # log_prob = (torch.log(torch.sigmoid(y_ui - y_uj))).mean()
    sig = torch.sigmoid(pos_pred - neg_pred) + 1e-7
    log_prob = torch.log(sig)
    loss = -log_prob
    return loss
