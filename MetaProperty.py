import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import math
import pickle
import gzip


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """
    def reset_parameters(self):
        """
        Initialize parameters.
        """
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class Attention(nn.Module):
    def __init__(self, input_size, embed_size, out_channels):
        super(Attention, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.out_channels = out_channels
        self.conv = nn.Conv1d(40, 40, 754)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5,5)),
            nn.Conv2d(1, self.out_channels, kernel_size=(1, self.embed_size - 4)),
            # nn.ReLU(),
            nn.MaxPool2d((self.input_size - 4, 1)))
        self.act = nn.ReLU()
        self.max_pool = nn.MaxPool2d((self.input_size, 1))

    def forward(self, review_emb, prop_scores, prop_pref):
        # Prop_scores reflect the properties of reviews
        # prop_scores shape: batch_num * num_property * num_reviews
        prop_scores = torch.einsum('imj,im->imj', prop_scores, prop_pref)
        review_emb = torch.einsum('ijk,imj->imjk', review_emb, prop_scores)
        review_emb = torch.mean(review_emb, 1)
        out = self.conv(review_emb)
        out = self.act(out)
        out = torch.reshape(out, (out.size(0), -1))
        return out    


class MetaProperty(nn.Module):

    def __init__(self, data_name):
        super(MetaProperty, self).__init__()
        self.input_size = 40
        self.embed_size = 768
        self.channels = 768
        self.latent_dim = 300
        self.review_out_dim = 600
        self.num_prop = 6
        if data_name == 'yelp':
            self.num_users = 47154
            self.num_items = 16531
        elif data_name == 'amazon':
            self.num_users = 26010
            self.num_items = 16514
        # self.user_item = pickle.load(gzip.open(root + 'user_item.p'))
        self.user_embedding = ScaledEmbedding(self.num_users + 1, self.review_out_dim)
        self.item_embedding = ScaledEmbedding(self.num_items + 1, self.review_out_dim)
        self.user_review_encode = Attention(self.input_size, self.embed_size, self.channels)
        self.item_review_encode = Attention(self.input_size, self.embed_size, self.channels)
        self.user_latent_vector = ScaledEmbedding(self.num_users + 1, self.latent_dim)
        self.item_latent_vector = ScaledEmbedding(self.num_items + 1, self.latent_dim)
        self.user_prop_pref = ScaledEmbedding(self.num_users + 1, self.num_prop)
        self.item_prop_pref = ScaledEmbedding(self.num_items + 1, self.num_prop)
        self.hidden_dim = 768
        self.output_dim = 768
        self.actLayer = nn.ReLU()
        self.user_bias = nn.Embedding(self.num_users + 1, 1)
        self.item_bias = nn.Embedding(self.num_items + 1, 1)
        self.mu_bias = Variable(torch.ones(1), requires_grad=True).cuda()

    def forward(self, user_id, item_id, user_review, user_review_scores, item_reviews, item_review_scores):
        # user
        user_feat = self.user_review_encode(user_review, user_review_scores, self.user_prop_pref(user_id))
        user_feat = user_feat.view(user_feat.size(0), -1)
        user_feat = self.actLayer(user_feat)
        for i in range(len(user_id)):
            self.user_embedding.weight.data[user_id[i]].copy_(user_feat[i])

        # item
        item_feat = self.item_review_encode(item_reviews, item_review_scores, self.item_prop_pref(item_id))
        item_feat = item_feat.view(item_feat.size(0), -1)
        item_feat = self.actLayer(item_feat)
        for i in range(len(item_id)):
            self.item_embedding.weight.data[item_id[i]].copy_(item_feat[i])
        
        out = torch.einsum('ij,ij->i', user_feat, item_feat) + self.user_bias(user_id).squeeze() + \
              self.item_bias(item_id).squeeze() + self.mu_bias
        return out
