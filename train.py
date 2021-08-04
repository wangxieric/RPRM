import torch
import torch.nn as nn
import sys
import data_loader
import MetaProperty
import math
import numpy as np
import pandas as pd
import pickle
import gzip
import time
from datetime import datetime
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from utils.loss import bpr_loss
from utils.evaluation import evaluate
from utils.sample import sample_items


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def limit(x):
    return (x - (torch.min(x))) / (torch.max(x) - torch.min(x)) + 1e-7

# Hyper Parameters
num_epochs = 30
batch_size = 512
learning_rate = 1e-3
negative_sample_size = 20
n_sample = 10
rank_task_weight = 0.9

print("batch_size: ", batch_size)
print("learning_rate: ", learning_rate)

data_name = 'amazon'
# in ['age', 'length', 'rate', 'polar_sentiment', 'helpful', 'polar_helpful']
review_property = 'all'

print("Loading data...", data_name)
print("Review property: ", review_property)
if data_name == 'yelp':
    num_users = 47154
    num_items = 16531
    train_loader, train_dataset = data_loader.get_loader('yelp_train.p', data_name, review_property, batch_size=batch_size)
    val_loader, val_dataset = data_loader.get_loader('yelp_valid.p', data_name, review_property, batch_size=batch_size, shuffle=False)
    test_loader, test_dataset = data_loader.get_loader('yelp_test.p', data_name, review_property, batch_size=batch_size, shuffle=False)
elif data_name == 'amazon':
    num_users = 26010
    num_items = 16514
    train_loader, train_dataset = data_loader.get_loader('amazon_train.p', data_name, review_property, batch_size=batch_size)
    val_loader, val_dataset = data_loader.get_loader('amazon_valid.p', data_name, review_property, batch_size=batch_size, shuffle=False)
    test_loader, test_dataset = data_loader.get_loader('amazon_test.p', data_name, review_property, batch_size=batch_size, shuffle=False)

print("train/val/test/: {:d}/{:d}/{:d}".format(len(train_loader), len(val_loader),len(test_loader)))
print("==================================================================================")

meta_property = MetaProperty.MetaProperty(data_name)


if torch.cuda.is_available():
    meta_property.cuda()

optimizer = torch.optim.Adam(meta_property.parameters(), lr=learning_rate)

print("==================================================================================")
print("Training Start..", datetime.now())
batch_loss = 0
batch_rat_loss = 0
batch_rank_loss = 0
# Train the Model
total_step = len(train_loader)
best_p1 = 0.0
best_p5 = 0.0
best_p10 = 0.0
best_r1 = 0.0
best_r5 = 0.0
best_r10 = 0.0
best_map = 0.0
# KL_loss = nn.KLDivLoss(reduction='batchmean')
for epoch in range(num_epochs):
    for idx, (user_id, item_id, user_review, user_review_scores, item_review, item_review_scores) in enumerate(train_loader):
        user_id = to_var(user_id)
        item_id = to_var(item_id)
        user_review = to_var(user_review)
        user_review_scores = to_var(user_review_scores)
        item_review = to_var(item_review)
        item_review_scores = to_var(item_review_scores)
        rank_pos_outputs = meta_property(user_id, item_id, user_review, user_review_scores,
                                         item_review, item_review_scores)
        for i in range(n_sample):
            sample_item_id, item_review, item_review_scores = train_dataset.get_neg_items(len(user_id))
            sample_item_id = to_var(torch.from_numpy(np.array(sample_item_id)))
            item_review = to_var(item_review)
            item_review_scores = to_var(item_review_scores)
            rank_neg_outputs = meta_property(user_id, sample_item_id, user_review, user_review_scores,
                                             item_review, item_review_scores)

            # Cos-UU loss
            # rank_loss = rank_loss + bpr_loss(rank_pos_outputs, rank_neg_outputs) + (1.0 / n_sample) *
            # torch.mean(cos(limit(meta_property.user_prop_pref(user_id)),
            # limit(meta_property.item_prop_pref(sample_item_id))))

            # Cos-UI loss
            # rank_loss = rank_loss + bpr_loss(rank_pos_outputs, rank_neg_outputs) + (1.0 / n_sample) * torch.mean(
            #    cos(limit(meta_property.item_prop_pref(item_id)), limit(meta_property.item_prop_pref(sample_item_id))))

            # KL-UU Loss
            # rank_loss = rank_loss + bpr_loss(rank_pos_outputs, rank_neg_outputs) - (1.0 / n_sample) * KL_loss(
            #     limit(meta_property.user_prop_pref(user_id)), limit(meta_property.item_prop_pref(sample_item_id)))

            # KL-UI Loss
            # rank_loss = rank_loss + bpr_loss(rank_pos_outputs, rank_neg_outputs) -
            # (1.0 / n_sample) * KL_loss(limit(meta_property.item_prop_pref(item_id)),
            # limit(meta_property.item_prop_pref(sample_item_id)))

            if i == 0:
                rank_loss = bpr_loss(rank_pos_outputs, rank_neg_outputs)
            else:
                rank_loss += bpr_loss(rank_pos_outputs, rank_neg_outputs)
        loss = rank_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(" loss: {}, time: {}".format(loss, datetime.now()))
    print("epoch: ", epoch, datetime.now())
    print("==================================================================================")
    print("Begin Ranking Prediction..")
    with torch.no_grad():
        k = [1, 5, 10]
        precision = [0.0] * len(k)
        recall = [0.0] * len(k)
        apks = 0.0
        precision_u = [[],[],[]]
        recall_u = [[],[],[]]
        for i, user in enumerate(range(num_users)):
            true_results = test_dataset.get_user_items(user)
            user_torch = to_var(torch.LongTensor([user]))
            user_emb = meta_property.user_embedding(user_torch)
            user_emb = user_emb.squeeze()
            user_lv = meta_property.user_latent_vector(user_torch).squeeze()
            user_emb = torch.cat((user_emb, user_lv))
            item_emb = meta_property.item_embedding.weight.data
            item_lv = meta_property.item_latent_vector.weight.data
            item_emb = torch.cat((item_emb, item_lv), 1)
            pred_long_term_results = torch.mv(item_emb, user_emb)
            item_bias = meta_property.item_bias(to_var(torch.LongTensor(range(num_items + 1)))).squeeze()
            pred_results = pred_long_term_results + meta_property.user_bias(user_torch).squeeze() + item_bias

            # remove visited items
            visited_items = train_dataset.get_user_items(user)
            pred_results[visited_items] = -np.inf
            pred_results[num_items] = -np.inf
            sub_precision, sub_recall, apk, sub_precision_u, sub_recall_u = evaluate(true_results, pred_results, ks=k)
            precision = [precision[i] + sub_precision[i] for i in range(len(k))]
            recall = [recall[i] + sub_recall[i] for i in range(len(k))]

            apks = apks + apk
            if i % 10000 == 0:
                print('Tested Users {0}: test result precision@1 {1:.5f}, recall@1 {2:.5f}, precision@5 {3:.5f},' 
                      'recall@5 {4:.5f}, precision@10 {5:.5f}, recall@10 {6:.5f}, MAP {7:.5f}, time: {8}'.format(i, precision[0] / num_users, 
                                                                                                        recall[0] / num_users, precision[1] / num_users,
                                                                                                        recall[1] / num_users, precision[2] / num_users, 
                                                                                                        recall[2] / num_users, apks / num_users, datetime.now()))
        print('Tested Users {0}: test result precision@1 {1:.5f}, recall@1 {2:.5f}, precision@5 {3:.5f},' 
                      'recall@5 {4:.5f}, precision@10 {5:.5f}, recall@10 {6:.5f}, MAP {7:.5f}, time: {8}'.format(i, precision[0] / num_users, 
                                                                                                        recall[0] / num_users, precision[1] / num_users,
                                                                                                        recall[1] / num_users, precision[2] / num_users, 
                                                                                                        recall[2] / num_users, apks / num_users, datetime.now()))
#         if epoch < 5:
#                 pickle.dump([precision, recall], gzip.open("epoch_{0}_results.p",'wb'))
        if (apks / num_users) > best_map:
            best_p1 = precision[0] / num_users 
            best_p5 = precision[1] / num_users
            best_p10 = precision[2] / num_users
            best_r1 = recall[0] / num_users
            best_r5 = recall[1] / num_users
            best_r10 = recall[2] / num_users
            best_map = apks / num_users

        print("==================================================================================")
        print("Testing End..")

print('Best Results after {0} epochs: test result precision@1 {1:.5f}, recall@1 {2:.5f}, precision@5 {3:.5f},' 
                  'recall@5 {4:.5f}, precision@10 {5:.5f}, recall@10 {6:.5f}, MAP {7:.5f}, time: {8}'.format(num_epochs, best_p1, best_r1, best_p5, best_r5,
                                                                                                             best_p10, best_r10, best_map, datetime.now()))
print("==================================================================================")
print("Training End..")

print("batch_size: ", batch_size)
print("learning_rate: ", learning_rate)
print("Loading data...", data_name)
print("Property: ", review_property)