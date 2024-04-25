import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE


class MyGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MyGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config["MyGCL"])
        self.n_layers = int(args["-n_layer"])
        self.cl_rate = float(args["-lambda"])
        self.temperature = float(args["-temperature"])
        self.prototype_num = int(args['-num_clusters'])
        self.proto_reg = float(args['-proto_reg'])
        self.model = MyGCL_Encoder(self.data, self.emb_size, self.n_layers , self.prototype_num)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb , user_prototypes , item_prototypes, all_embddings = model()
                user_emb, pos_item_emb, neg_item_emb = (
                    rec_user_emb[user_idx],
                    rec_item_emb[pos_idx],
                    rec_item_emb[neg_idx],
                )
                BPR_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                L2_reg_loss = (
                    l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                    / self.batch_size
                )
                CL_loss = self.cl_rate * self.cal_cl_loss(
                    all_embddings[0], all_embddings[2], [user_idx, pos_idx]
                )
                Proto_loss = self.proto_reg * self.cal_proto_loss(user_emb , pos_item_emb , user_prototypes , item_prototypes)
                batch_loss = BPR_loss + L2_reg_loss + CL_loss + Proto_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print(
                        "training:",
                        epoch + 1,
                        "batch",
                        n,
                        "BPR_loss:",
                        BPR_loss.item(),
                        "CL_loss:",
                        CL_loss.item(),
                        "Proto_loss:" ,
                        Proto_loss.item(),
                        "batch_loss:",
                        batch_loss.item(),
                    )
            with torch.no_grad():
                self.user_emb, self.item_emb, _ , _ , _ = model()
            # if epoch % 5 == 0:
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, view1, view2, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view1, item_view1 = torch.split(
            view1, [self.data.user_num, self.data.item_num]
        )
        user_view2, item_view2 = torch.split(
            view2, [self.data.user_num, self.data.item_num]
        )
        user_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temperature)
        item_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temperature)
        return user_loss + item_loss

    def cal_proto_loss(self , user_embed , item_embed , user_prototypes , item_prototypes) :
        user_loss = self.proto_loss(user_embed , user_prototypes , self.temperature) 
        item_loss = self.proto_loss(item_embed , item_prototypes , self.temperature)
        return user_loss + item_loss

    def proto_loss(self, z, prototypes, temperature=0.1):
        # Compute scores between embeddings and prototypes
        # 3862x64 and 2000x64
        scores = torch.mm(z, prototypes.T)
        score_s = scores
        # score_t = scores[: z.size(0) // 2]
        # score_s = scores[z.size(0) // 2 :]

        # Apply the Sinkhorn-Knopp algorithm to get soft cluster assignments
        # q_t = self.sinkhorn_knopp(score_t)
        q_s = self.sinkhorn_knopp(score_s)

        # log_p_t = torch.log_softmax(score_t / temperature + 1e-7, dim=1)
        log_p_s = torch.log_softmax(score_s / temperature + 1e-7, dim=1)

        # Calculate cross-entropy loss
        # loss_t = torch.mean(
        #     -torch.sum(
        #         q_s * log_p_t,
        #         dim=1,
        #     )
        # )
        loss_s = torch.mean(
            -torch.sum(
                q_s * log_p_s,# q_t * log_p_s,
                dim=1,
            )
        )
        # proto loss is the average of loss_t and loss_s
        # proto_loss = (loss_t + loss_s) / 2
        return loss_s
    
    def sinkhorn_knopp(self, scores, epsilon=0.05, n_iters=3):
        with torch.no_grad():
            scores_max = torch.max(scores, dim=1, keepdim=True).values
            scores_stable = scores - scores_max
            Q = torch.exp(scores_stable / epsilon).t()
            Q /= Q.sum(dim=1, keepdim=True) + 1e-8

            K, B = Q.shape
            u = torch.zeros(K).to(scores.device)
            r = torch.ones(K).to(scores.device) / K
            c = torch.ones(B).to(scores.device) / B

            for _ in range(n_iters):
                u = Q.sum(dim=1)
                Q *= (r / (u + 1e-8)).unsqueeze(1)
                Q *= (c / Q.sum(dim=0)).unsqueeze(0)

            Q = (Q / Q.sum(dim=0, keepdim=True)).t()
            return Q

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _ , _ , _= self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class MyGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers , prototype_num):
        super(MyGCL_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(
            self.norm_adj
        ).cuda()
        self.prototype_num = prototype_num
        self.prototypes_dict = self._init_prototypes()

    def _init_prototypes(self):
        initializer = nn.init.xavier_uniform_
        prototypes_dict = nn.ParameterDict(
            {
                "user_prototypes": nn.Parameter(
                    initializer(torch.empty(self.prototype_num, self.latent_size))
                ),
                "item_prototypes": nn.Parameter(
                    initializer(torch.empty(self.prototype_num, self.latent_size))
                ),
            }
        )
        return prototypes_dict

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict(
            {
                "user_emb": nn.Parameter(
                    initializer(torch.empty(self.data.user_num, self.latent_size))
                ),
                "item_emb": nn.Parameter(
                    initializer(torch.empty(self.data.item_num, self.latent_size))
                ),
            }
        )
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat(
            [self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], 0
        )
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        light_gcn_embedding = torch.stack(all_embeddings, dim=1)
        light_gcn_embedding = torch.mean(light_gcn_embedding, dim=1)
        user_all_embeddings = light_gcn_embedding[: self.data.user_num]
        item_all_embeddings = light_gcn_embedding[self.data.user_num :]
        return user_all_embeddings, item_all_embeddings , self.prototypes_dict['user_prototypes'] , self.prototypes_dict['item_prototypes'], all_embeddings


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out
