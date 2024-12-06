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
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20

class DiffNetpp(GraphRecommender):
    
    def __init__(self, conf, training_set, valid_set, test_set, **kwargs):
        super(DiffNetpp, self).__init__(conf, training_set, valid_set, test_set, **kwargs)
        self.n_layers = int(self.config['n_layer'])
        self.lambd = conf['lambda']
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        S = self.social_data.get_social_mat()
        self.model = DiffNetpp_Encoder(self.data, self.emb_size, self.n_layers, S)

    
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


            # valid
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                stop = self.fast_evaluation_valid(epoch, model, 'pt/'+self.config['model.name']+'_best.pt')
                if stop:
                    break

        # test
        best_model = torch.load('pt/'+self.config['model.name']+'_best.pt')
        with torch.no_grad():
            self.user_emb, self.item_emb = best_model()
        measure = self.fast_evaluation_test(epoch)
        file = open('model_result/' + self.config['data'] + '/' + self.config['model.name'] + '_' + str(self.config['times']) + '.pkl', 'wb')
        pickle.dump(measure, file)
        file.close()


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    
    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.detach().cpu().numpy()

    __classcell__ = None


class MLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.normal_(self.fc1.weight, mean = 0, std = 0.01)
        torch.nn.init.normal_(self.fc2.weight, mean = 0, std = 0.01)

    
    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        return out




class DiffConv(nn.Module):
    
    def __init__(self = None):
        super(DiffConv, self).__init__()

    
    def forward(self, R, X):
        side_embeddings = torch.matmul(R, X)
        return side_embeddings



class DiffNetpp_Encoder(nn.Module):
    
    def __init__(self, data, emb_size, n_layers, S):
        super(DiffNetpp_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        R = data.interaction_mat
        self.R = TorchGraphInterface.convert_sparse_mat_to_tensor(R).cuda()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.sparse_S = TorchGraphInterface.convert_sparse_mat_to_tensor(S).cuda()
        self.relu = nn.ReLU(inplace = True)
        self.conv_layers = nn.ModuleList()
        for i in range(self.layers):
            layer = DiffConv().cuda()
            self.conv_layers.append(layer)
        self.mlp1 = MLP(self.latent_size * 2, self.latent_size, 1).cuda()
        self.mlp2 = MLP(self.latent_size * 2, self.latent_size, 1).cuda()
        self.mlp3 = MLP(self.latent_size * 2, self.latent_size, 1).cuda()
        self.mlp4 = MLP(self.latent_size * 2, self.latent_size, 1).cuda()

    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))) })
        return embedding_dict

    
    def attention_weight(self, graph, src_embedd, trg_embedd):
        '''
        '''
        edges = graph._indices()
        src = src_embedd[edges[0], :]
        trg = trg_embedd[edges[1], :]
        concate_embedd = torch.cat([src,trg], dim = -1)
        return concate_embedd

    
    def forward(self):
        U = self.embedding_dict['user_emb']
        V = self.embedding_dict['item_emb']
        E = torch.cat([U,V], dim = 0)
        v_concate = self.attention_weight(self.R.t(), V, U)
        v_weights = self.mlp1(v_concate).reshape(-1)
        u_concate = self.attention_weight(self.R, U, V)
        u_weights = self.mlp3(u_concate).reshape(-1)
        s_concate = self.attention_weight(self.sparse_S, U, U)
        s_weights = self.mlp2(s_concate).reshape(-1)
        S_weight = torch.sparse.FloatTensor(self.sparse_S._indices(), s_weights, self.sparse_S.shape)
        S_weight = torch.sparse.softmax(S_weight, dim = 1)
        A_weights = torch.cat([u_weights, v_weights], dim = 0)
        A_hat_weight = torch.sparse.FloatTensor(self.sparse_norm_adj._indices(), A_weights, self.sparse_norm_adj.shape)
        A_hat_weight = torch.sparse.softmax(A_hat_weight, dim = 1)
        S = torch.mul(self.sparse_S, S_weight).to_sparse()
        A_hat = torch.mul(self.sparse_norm_adj, A_hat_weight)
        concate_g_embeddings = E.clone()
        for i in range(self.layers):
            U, V = torch.split(E, [self.data.user_num,self.data.item_num])
            U_s = torch.sparse.mm(S, U)
            U_r = torch.sparse.mm(self.R, V)
            concate_u_e1 = torch.cat([U,U_s], dim = -1)
            gamma_a1 = self.mlp4(concate_u_e1).reshape(-1)
            gamma_a1 = torch.softmax(gamma_a1, dim = 0).reshape(-1, 1)
            concate_u_e2 = torch.cat([U,U_r], dim = -1)
            gamma_a2 = self.mlp4(concate_u_e2).reshape(-1)
            gamma_a2 = torch.softmax(gamma_a2, dim = 0).reshape(-1, 1)
            ones = torch.ones(self.data.item_num).reshape(-1, 1).cuda()
            gamma_a2 = torch.cat([gamma_a2,ones], dim = 0)
            E = torch.cat([U,V], dim = 0)
            E = self.conv_layers[i](A_hat, E.mul(gamma_a2))
            U_out, V_out = torch.split(E, [self.data.user_num,self.data.item_num])
            U = U_out + U_s.mul(gamma_a1)
            V = V_out + V
            E = torch.cat([U,V], dim = 0)
            concate_g_embeddings = torch.cat([concate_g_embeddings,E], dim = 1)
        user_all_embeddings = concate_g_embeddings[:self.data.user_num]
        item_all_embeddings = concate_g_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


