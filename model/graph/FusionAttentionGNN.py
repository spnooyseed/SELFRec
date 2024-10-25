

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import itertools
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from scipy.sparse import coo_matrix
from time import time
from base.graph_recommender import GraphRecommender


import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22


class FusionAttentionGNN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(FusionAttentionGNN, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['FusionAttentionGNN'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.att_topk = int(args['-att_topk'])
        self.model = FusionAttentionGNN_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.att_topk)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, att_user_emb , att_item_emb, fusion_user_emb, fusion_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                # user_emb, pos_item_emb, neg_item_emb = fusion_user_emb[user_idx], fusion_item_emb[pos_idx], fusion_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cal_cl_loss([user_idx,pos_idx], fusion_user_emb , fusion_item_emb,rec_user_emb, rec_item_emb)
                # cl_loss += self.cal_cl_loss([user_idx,pos_idx], fusion_user_emb , fusion_item_emb, att_user_emb , att_item_emb)
                cl_loss *= self.cl_rate
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb , _ , _ , _ , _= self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_fusion , item_fusion , user_cl , item_cl):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_fusion[u_idx], user_cl[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_fusion[i_idx], item_cl[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb,_ , _ , _ , _ = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class FusionAttentionGNN_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, att_topk):
        super(FusionAttentionGNN_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.topK = att_topk
        self.f_w = nn.Linear(self.emb_size , self.emb_size)
        self.attention = SampleAttention(self.emb_size , 0.2)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        # GNN
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        # Attention
        sim = torch.matmul(ego_embeddings , ego_embeddings.t())
        # sim = sim + 0.5 * torch.sparse.mm(self.sparse_norm_adj , sim)
        topk = sim.topk(self.topK).indices
        sample_embeddings = ego_embeddings[topk]
        atten_embed = self.attention(ego_embeddings , sample_embeddings)
        att_user_emb, att_item_emb = torch.split(atten_embed, [self.data.user_num, self.data.item_num])

        # fusion
        a1 = torch.tanh(self.f_w(all_embeddings))
        a2 = torch.tanh(self.f_w(atten_embed))
        a3 = torch.stack([a1,a2])
        a4 = torch.softmax(a3 , dim=1)
        fusion_emb = a4[0,:] * all_embeddings + a4[1,:] * atten_embed
        fusion_user_emb, fusion_item_emb = torch.split(fusion_emb, [self.data.user_num, self.data.item_num])

        return user_all_embeddings, item_all_embeddings , att_user_emb , att_item_emb , fusion_user_emb , fusion_item_emb


class SampleAttention(nn.Module):
    def __init__(self , dim, dropout):
        super(SampleAttention , self).__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim , dim)
        self.w_k = nn.Linear(dim , dim)
        self.w_v = nn.Linear(dim , dim)
        torch.nn.init.xavier_normal(self.w_q.weight, gain=1)
        torch.nn.init.xavier_normal(self.w_k.weight, gain=1)
        torch.nn.init.xavier_normal(self.w_v.weight, gain=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self , Q , K_V):
        query , key , val = self.w_q(Q) , self.w_k(K_V) , self.w_v(K_V)
        tmp1 = torch.matmul(query.unsqueeze(1) , key.transpose(1,2)) # QK^T
        tmp2 = tmp1 / torch.tensor(self.dim).sqrt()
        tmp3 = torch.softmax(tmp2,dim=-1)
        tmp4 = torch.matmul(tmp3 , val).squeeze(1)

        y = self.dropout(tmp4) + 0.1 * Q
        return y