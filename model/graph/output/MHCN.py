import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from data.social import Relation
import torch.nn.functional as F
import pickle
import time

class MHCN(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set, **kwargs):
        super(MHCN, self).__init__(conf, training_set, valid_set, test_set, **kwargs)
        self.n_layers = int(self.config['n_layer'])
        self.lambd = conf['lambda']
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        S = self.social_data.get_social_mat()
        self.model = MHCN_Encoder(self.data, self.emb_size, self.n_layers, S, self.social_data)

    
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                neg_item_emb = rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                    continue
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                stop = self.fast_evaluation_valid(epoch, model, 'pt/' + self.config['model.name'] + '_best.pt')
                if stop:
                    break
                
                best_model = torch.load('pt/' + self.config['model.name'] + '_best.pt')
                with torch.no_grad():
                    self.user_emb, self.item_emb = best_model()

        measure = self.fast_evaluation_test(epoch)

    
    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    
    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.detach().cpu().numpy()



class GatingLayer(nn.Module):
    def __init__(self, dim):
        super(GatingLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(self.dim, self.dim)
        self.activation = nn.Sigmoid()

    def forward(self, emb):
        embedding = self.linear(emb)
        embedding = self.activation(embedding)
        embedding = torch.mul(emb, embedding)
        return embedding

class AttLayer(nn.Module):
    def __init__(self, dim):
        super(AttLayer, self).__init__()
        self.dim = dim
        self.attention_mat = nn.Parameter(torch.randn([self.dim, self.dim]))
        self.attention = nn.Parameter(torch.randn([1, self.dim]))

    def forward(self, *embs):
        weights = []
        emb_list = []
        for embedding in embs:
            weights.append(torch.sum(torch.mul(self.attention, torch.matmul(embedding, self.attention_mat)), dim=1))
            emb_list.append(embedding)
        score = torch.nn.Softmax(dim=0)(torch.stack(weights, dim=0))
        embeddings = torch.stack(emb_list, dim=0)
        mixed_embeddings = torch.mul(embeddings, score.unsqueeze(dim=2).repeat(1, 1, self.dim)).sum(dim=0)
        return mixed_embeddings



class MHCN_Encoder(nn.Module):
    
    def __init__(self, data, emb_size, n_layers, S, social_data):
        super(MHCN_Encoder, self).__init__()
        self.data = data
        self.social_data = social_data
        self.latent_size = emb_size
        self.layers = n_layers
        self.interact_adj = data.interaction_mat
        self.norm_adj = data.norm_interact
        self.embedding_dict = self._init_model()
        self.sparse_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.interact_adj).cuda()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.sparse_S = TorchGraphInterface.convert_sparse_mat_to_tensor(S).cuda()
        self.H_s, self.H_j, self.H_p = self.get_motif_adj_matrix(self.sparse_S, self.sparse_adj)
        self.gating_c1 = GatingLayer(self.latent_size)
        self.gating_c2 = GatingLayer(self.latent_size)
        self.gating_c3 = GatingLayer(self.latent_size)
        self.gating_simple = GatingLayer(self.latent_size)
        self.attention_layer = AttLayer(self.latent_size)

    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))) })
        return embedding_dict

    
    def get_motif_adj_matrix(self, S, R):
        Y = R
        B = S.mul(S.t())
        U = S - B
        C1 = (torch.sparse.mm(U, (U))).mul(U.t())
        A1 = C1 + C1.t()
        C2 = (torch.sparse.mm(B, (U))).mul(U.t()) + (torch.sparse.mm(U, (B))).mul(U.t()) + (torch.sparse.mm(U, (U))).mul(B)
        A2 = C2 + C2.t()
        C3 = (torch.sparse.mm(B, (B))).mul(U) + (torch.sparse.mm(B, (U))).mul(B) + (torch.sparse.mm(U, (B))).mul(B)
        A3 = C3 + C3.t()
        A4 = (torch.sparse.mm(B, (B))).mul(B)
        C5 = (torch.sparse.mm(U, (U))).mul(U) + (torch.sparse.mm(U, (U.t()))).mul(U) + (torch.sparse.mm(U.t(), (U))).mul(U)
        A5 = C5 + C5.t()
        A6 = (torch.sparse.mm(U, (B))).mul(U) + (torch.sparse.mm(B, (U.t()))).mul(U.t()) + (torch.sparse.mm(U.t(), (U))).mul(B)
        A7 = (torch.sparse.mm(U.t(), (B))).mul(U.t()) + (torch.sparse.mm(B, (U))).mul(U) + (torch.sparse.mm(U, (U.t()))).mul(B)
        A8 = (torch.sparse.mm(Y, (Y.t()))).mul(B)
        A9 = (torch.sparse.mm(Y, (Y.t()))).mul(U)
        A9 = A9 + A9.t()
        A10  = torch.sparse.mm(Y, (Y.t())) - A8 - A9

        H_s = (A1 + A2 + A3 + A4 + A5 + A6 + A7)
        H_j = (A8 + A9)
        H_p = A10

        # add epsilon to avoid divide by zero Warning
        D_s = torch.sparse.sum(H_s, dim=1).to_dense()
        D_s[D_s == 0] = 1e-10
        Ds_inv = D_s.pow(-1)
        Ds_inv_sparse = torch.diag(Ds_inv)
        H_s = torch.sparse.mm(Ds_inv_sparse, H_s).to_sparse()

        D_j = torch.sparse.sum(H_j, dim=1).to_dense()
        D_j[D_j == 0] = 1e-10
        Dj_inv = D_j.pow(-1)
        Dj_inv_sparse = torch.diag(Dj_inv)
        H_j = torch.sparse.mm(Dj_inv_sparse, H_j).to_sparse()

        D_p = torch.sparse.sum(H_p, dim=1).to_dense()
        D_p[D_p == 0] = 1e-10
        Dp_inv = D_p.pow(-1)
        Dp_inv_sparse = torch.diag(Dp_inv)
        H_p = torch.sparse.mm(Dp_inv_sparse, H_p).to_sparse()

        return H_s, H_j, H_p

    
    def forward(self):
        user_embeddings = self.embedding_dict['user_emb']
        item_embeddings = self.embedding_dict['item_emb']
        user_embeddings_c1 = self.gating_c1(user_embeddings)
        user_embeddings_c2 = self.gating_c2(user_embeddings)
        user_embeddings_c3 = self.gating_c3(user_embeddings)
        simple_user_embeddings = self.gating_simple(user_embeddings)
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        all_embeddings_i = [item_embeddings]
        for layer_idx in range(self.layers):
            mixed_embedding = self.attention_layer(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3) + simple_user_embeddings / 2
            user_embeddings_c1 = torch.sparse.mm(self.H_s, user_embeddings_c1)
            norm_embeddings = F.normalize(user_embeddings_c1, p = 2, dim = 1)
            all_embeddings_c1 += [
                norm_embeddings]
            user_embeddings_c2 = torch.sparse.mm(self.H_j, user_embeddings_c2)
            norm_embeddings = F.normalize(user_embeddings_c2, p = 2, dim = 1)
            all_embeddings_c2 += [
                norm_embeddings]
            user_embeddings_c3 = torch.sparse.mm(self.H_p, user_embeddings_c3)
            norm_embeddings = F.normalize(user_embeddings_c3, p = 2, dim = 1)
            all_embeddings_c3 += [
                norm_embeddings]
            new_item_embeddings = torch.sparse.mm(self.sparse_norm_adj.t(), mixed_embedding)
            norm_embeddings = F.normalize(new_item_embeddings, p = 2, dim = 1)
            all_embeddings_i += [
                norm_embeddings]
            simple_user_embeddings = torch.sparse.mm(self.sparse_norm_adj, item_embeddings)
            norm_embeddings = F.normalize(simple_user_embeddings, p = 2, dim = 1)
            all_embeddings_simple += [
                norm_embeddings]
            item_embeddings = new_item_embeddings
        user_embeddings_c1 = torch.stack(all_embeddings_c1, dim = 0).sum(dim = 0)
        user_embeddings_c2 = torch.stack(all_embeddings_c2, dim = 0).sum(dim = 0)
        user_embeddings_c3 = torch.stack(all_embeddings_c3, dim = 0).sum(dim = 0)
        simple_user_embeddings = torch.stack(all_embeddings_simple, dim = 0).sum(dim = 0)
        item_all_embeddings = torch.stack(all_embeddings_i, dim = 0).sum(dim = 0)
        user_all_embeddings = self.attention_layer(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
        user_all_embeddings += simple_user_embeddings / 2
        return user_all_embeddings, item_all_embeddings

