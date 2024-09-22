import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22

from tqdm import tqdm

class AttentionGNN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(AttentionGNN, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['AttentionGNN'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = AttentionGNN_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.config['training.set'])

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in tqdm(range(self.maxEpoch),'epoch'):
            for n, batch in tqdm(enumerate(next_batch_pairwise(self.data, self.batch_size)) , 'epoch'):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

import os
base_dir = os.path.join(os.path.dirname(__file__) , '../../')
class AttentionGNN_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers,data_dir):
        super(AttentionGNN_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()

        sparse_norm_adj_dir = os.path.join(os.path.dirname(data_dir) , 'sparse_norm_adj')
        if os.path.exists(sparse_norm_adj_dir) :
            self.sparse_norm_adj = torch.load(sparse_norm_adj_dir)
        else :
            self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
            torch.save(self.sparse_norm_adj,sparse_norm_adj_dir)

        sparse_adj_dir = os.path.join(os.path.dirname(data_dir) , 'sparse_adj')
        if os.path.exists(sparse_adj_dir) :
            self.sparse_adj = torch.load(sparse_adj_dir)
        else :
            self.sparse_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.ui_adj).cuda()
            torch.save(self.sparse_adj,sparse_adj_dir)

        self.TopK = 5
        self.attention = SampleAttention(self.emb_size)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_gnn_embeddings = []
        for k in range(self.n_layers):
            gnn_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(gnn_embeddings).cuda()
                gnn_embeddings += torch.sign(gnn_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_gnn_embeddings.append(ego_embeddings)
            import pdb
            pdb.set_trace()
            sim = torch.matmul(ego_embeddings , ego_embeddings.t())
            sim = sim + 0.5 * torch.sparse.mm(self.sparse_adj , sim)
            topk = sim.topk(self.TopK).indices
            sample_embeddings = ego_embeddings[topk]

            atten_embed = self.attention(ego_embeddings , sample_embeddings)
        all_gnn_embeddings = torch.stack(all_gnn_embeddings, dim=1)
        all_gnn_embeddings = torch.mean(all_gnn_embeddings, dim=1)
        user_gnn_embeddings, item_gnn_embeddings = torch.split(all_gnn_embeddings, [self.data.user_num, self.data.item_num])
        return user_gnn_embeddings, item_gnn_embeddings

class SampleAttention(nn.Module):
    def __init__(self , dim):
        super(SampleAttention , self).__init__()
        self.dim = dim
        self.w_q = nn.Linear(dim , 4*dim)
        self.w_k = nn.Linear(dim , 4*dim)
        self.w_v = nn.Linear(dim , 4*dim)
        torch.nn.init.xavier_normal(self.w_q.weight, gain=1)
        torch.nn.init.xavier_normal(self.w_k.weight, gain=1)
        torch.nn.init.xavier_normal(self.w_v.weight, gain=1)

    def forward(self , Q , K_V):
        query , key , val = self.w_q(Q) , self.w_k(K_V) , self.w_v(K_V)
        import pdb
        pdb.set_trace()
        key.transpose(1,2)


