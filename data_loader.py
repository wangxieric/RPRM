import torch
from torch.utils import data
import pickle
import gzip
import numpy as np
from torch.nn import CosineSimilarity
from torch.utils.data.dataloader import default_collate
import torch
from datetime import datetime
import random 
import torch.nn.functional as F

class ReviewDataset(data.Dataset):
    def __init__(self, pickle_path, data_name, review_property):
        if data_name == 'yelp':
            root = '../../data/yelp/' 
            self.num_users = 47154
            self.num_items = 16531
        elif data_name == 'amazon': 
            root = '../../data/amazon/'
            self.num_users = 26010
            self.num_items = 16514
        self.dataset = pickle.load(gzip.open(root + pickle_path, 'rb'))
        self.dataset.reset_index()
        self.users = list(self.dataset['user_id'])
        self.user_dataset_group = self.dataset.groupby('user_id')

        # load user-related information
        self.user_item = pickle.load(gzip.open(root + 'user_item.p'))
        self.user_review = pickle.load(gzip.open(root + 'user_review.p'))
        self.user_review_age = pickle.load(gzip.open(root + 'user_review_age.p'))
        self.user_review_helpful = pickle.load(gzip.open(root + 'user_review_helpful.p'))
        self.user_review_polar_helpful = pickle.load(gzip.open(root + 'user_review_polar_helpful.p'))
        self.user_review_length = pickle.load(gzip.open(root + 'user_review_length.p'))
        self.user_review_rate = pickle.load(gzip.open(root + 'user_review_rate.p'))
        self.user_review_polar_senti = pickle.load(gzip.open(root + 'user_review_polar_senti.p'))
        self.user_review_score_dict = [self.user_review_age,
                                       self.user_review_helpful,
                                       self.user_review_polar_helpful,
                                       self.user_review_length,
                                       self.user_review_rate,
                                       self.user_review_polar_senti]
        
        # load item-related information
        self.item_user = pickle.load(gzip.open(root + 'item_user.p'))
        self.item_review = pickle.load(gzip.open(root + 'item_review.p'))
        self.item_review_age = pickle.load(gzip.open(root + 'item_review_age.p'))
        self.item_review_helpful = pickle.load(gzip.open(root + 'item_review_helpful.p'))
        self.item_review_polar_helpful = pickle.load(gzip.open(root + 'item_review_polar_helpful.p'))
        self.item_review_length = pickle.load(gzip.open(root + 'item_review_length.p'))
        self.item_review_rate = pickle.load(gzip.open(root + 'item_review_rate.p'))
        self.item_review_polar_senti = pickle.load(gzip.open(root + 'user_review_polar_senti.p'))
        self.item_review_score_dict = [self.item_review_age,
                                       self.item_review_helpful,
                                       self.item_review_polar_helpful,
                                       self.item_review_length,
                                       self.item_review_rate,
                                       self.item_review_polar_senti]
        
        self.mode = 'pos'
        self.word_dim = 768
        self.max_reviews = 40
        self.rating_prediction = True
        self.review_property = review_property
#         if review_property == 'age':
#             self.user_review_scores = self.user_review_age
#             self.item_review_scores = self.item_review_age
#         elif review_property == 'helpful':
#             self.user_review_scores = self.user_review_helpful
#             self.item_review_scores = self.item_review_helpful
#         elif review_property == 'polar_helpful':
#             self.user_review_scores = self.user_review_polar_helpful
#             self.item_review_scores = self.item_review_polar_helpful
#         elif review_property == 'length':
#             self.user_review_scores = self.user_review_length
#             self.item_review_scores = self.item_review_length
#         elif review_property == 'rate':
#             self.user_review_scores = self.user_review_rate
#             self.item_review_scores = self.item_review_rate
#         elif review_property == 'polar_sentiment':
#             self.user_review_scores = self.user_review_polar_senti
#             self.item_review_scores = self.item_review_polar_senti

    def __getitem__(self, idx):
        # loading user data 
        # resulted user data shape (batch_size, max_reviews * word_dim)
        row = self.dataset.iloc[idx]
        user_id = row['user_id']
        item_id = row['business_id']
        
        user_review = torch.from_numpy(np.array(self.get_user_review(user_id))).float()
        user_review_scores = torch.from_numpy(np.array(self.get_user_review_scores(user_id))).float()

        item_review = torch.from_numpy(np.array(self.get_item_review(item_id) if item_id in self.item_review 
                                                  else np.zeros((self.max_reviews, self.word_dim)))).float()
        
        item_review_scores = torch.from_numpy(np.array(self.get_item_review_scores(item_id))).float()
        
        return (user_id, item_id, user_review, user_review_scores, item_review, item_review_scores)

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.dataset)

    def limit(self, x):
        return (x - (torch.min(x))) / (torch.max(x) - torch.min(x))

    # KL-Sampling
    # def get_neg_item_ids(self, item_id, model):
    #     sample_ids = []
    #     for item in item_id:
    #         item_prop = model.item_prop_pref(item).unsqueeze(0).repeat(self.num_items + 1, 1)
    #         sim_score = (1 - self.limit(F.kl_div(item_prop, model.item_prop_pref.weight.data,
    #         reduction="none").sum(dim=1)))
    #         # remove the positive item
    #         sim_score[item] = 0.0
    #         score = (sim_score + 1) / 2.0
    #         score = score.data.cpu().numpy()
    #         score /= score.sum().astype(float)
    #         sample_ids.append(np.random.choice(self.num_items + 1, 10, p=score, replace=False).tolist())
    #     sample_ids = np.array(sample_ids)
    #     sample_ids = sample_ids.reshape(10, -1)
    #     return sample_ids

    # Cosine Sampling
    # def get_neg_item_ids(self, item_id, model):
    #     sample_ids = []
    #     cos = CosineSimilarity(dim=1, eps=1e-7)
    #     for item in item_id:
    #         item_prop = model.item_prop_pref(item).unsqueeze(0).repeat(self.num_items + 1, 1)
    #         score = (cos(item_prop, model.item_prop_pref.weight.data) + 1) / 2.0
    #         score = score.data.cpu().numpy()
    #         score /= score.sum().astype(float)
    #         sample_ids.append(np.random.choice(self.num_items + 1, 10, p=score, replace=False).tolist())
    #     sample_ids = np.array(sample_ids)
    #     sample_ids = sample_ids.reshape(10, -1)
    #     return sample_ids

    def get_neg_items(self, batch_size):
        # basic negative sampling
        item_ids = random.sample(range(self.num_items), batch_size)
        
        item_review = torch.from_numpy(np.array([self.get_item_review(item_id) if item_id in self.item_review 
                                                 else np.zeros((self.max_reviews, self.word_dim)) 
                                                 for item_id in item_ids])).float()
        
        item_review_scores = torch.from_numpy(np.array([self.get_item_review_scores(item_id) for item_id in item_ids])).float()
        
        return (item_ids, item_review, item_review_scores)
    
    def get_user_review(self, user_id):
        if user_id in self.user_review:
            user_reviews = self.user_review[user_id]
            review_len = len(user_reviews)
            if review_len < self.max_reviews:
                pad_len = self.max_reviews - review_len
                pad_vector = np.zeros((pad_len, self.word_dim))
                user_reviews = np.concatenate((user_reviews, pad_vector), axis=0)
            else:
                user_reviews = user_reviews[:self.max_reviews]
        else:
            user_reviews = np.zeros((self.max_reviews, self.word_dim))
        return user_reviews
    
    def get_user_review_scores(self, user_id):
        user_review_scores = []
        for score_dict in self.user_review_score_dict:
            if user_id in score_dict:
                user_review_score = score_dict[user_id]
                scores_len = len(user_review_score)
                if scores_len < self.max_reviews:
                    pad_len = self.max_reviews - scores_len
                    user_review_score = np.concatenate((user_review_score, np.asarray([0.0] * pad_len)))
                else:
                    user_review_score = user_review_score[:self.max_reviews]
            else:
                user_review_score = np.zeros(self.max_reviews)
            user_review_scores.append(user_review_score)
        user_review_scores = np.array(user_review_scores)
        return user_review_scores
    
    def get_item_review(self, item_id):
        item_reviews = self.item_review[item_id]
        review_len = len(item_reviews)
        if review_len < self.max_reviews:
            pad_len = self.max_reviews - review_len
            pad_vector = np.zeros((pad_len, self.word_dim))
            item_reviews = np.concatenate((item_reviews, pad_vector), axis=0)
        else:
            item_reviews = item_reviews[:self.max_reviews]
        return item_reviews
    
    def get_item_review_scores(self, item_id):
        item_review_scores = []
        for score_dict in self.item_review_score_dict:
            if item_id in score_dict:
                item_review_score = score_dict[item_id]
                score_len = len(item_review_score)
                if score_len < self.max_reviews:
                    pad_len = self.max_reviews - score_len
                    item_review_score = np.concatenate((item_review_score, np.asarray([0.0] * pad_len)))
                else:
                    item_review_score = item_review_score[:self.max_reviews]
            else:
                item_review_score = np.zeros(self.max_reviews)
            item_review_scores.append(item_review_score)
        item_review_scores = np.array(item_review_scores)
        return item_review_scores
    
    def get_user_items(self, user_id):
        df = self.user_dataset_group.get_group(user_id)
        user_items = list(df['business_id'])
        return user_items
    
    def pad_data(self, user_item_list, max_len, pad_id):
        result_list = np.zeros((len(user_item_list), max_len))
        result_list[:] = pad_id
        # left pad
        for i in range(len(result_list)):
            result_list[i, (max_len-len(user_item_list[i])):] = user_item_list[i]
        result_list = result_list.astype(int)
        return result_list
        
        
def my_collate(batch):

    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def get_loader(data_path, data_name, review_property, batch_size=100, shuffle=True, num_workers=1):
    """Builds and returns Dataloader."""
    dataset = ReviewDataset(data_path, data_name, review_property)
    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate)
    return data_loader, dataset
