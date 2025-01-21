import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from data.social import Relation
import torch.nn.functional as F
import pickle

class DiffNet(GraphRecommender):
    
    def __init__(self, conf, training_set, valid_set, test_set, **kwargs):
        super(DiffNet, self).__init__(conf, training_set, valid_set, test_set, **kwargs)
        self.n_layers = int(self.config['n_layer'])
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        S = self.social_data.get_social_mat()
        sparse_S = self.social_data.convert_to_laplacian_mat(S)
        self.model = DiffNet_Encoder(self.data, self.emb_size, self.n_layers, sparse_S)

    
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



class DiffNet_Encoder(nn.Module):
    
    def __init__(self, data, emb_size, n_layers, sparse_S):
        super(DiffNet_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.interaction_mat
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.sparse_S = TorchGraphInterface.convert_sparse_mat_to_tensor(sparse_S).cuda()
        self.relu = nn.ReLU(inplace = True)
        self.diff_layers = nn.ModuleList()
        for i in range(self.layers):
            self.diff_layers.append(nn.Linear(self.latent_size * 2, self.latent_size, bias = True))
        for i in range(self.layers):
            nn.init.xavier_uniform_(self.diff_layers[i].weight)

    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))) })
        return embedding_dict

    
    def forward(self):
        U = self.embedding_dict['user_emb']
        V = self.embedding_dict['item_emb']
        for i in range(self.layers):
            U = torch.concat([
                torch.sparse.mm(self.sparse_S, U),
                U], dim = 1)
            U = self.diff_layers[i](U)
            U = self.relu(U)
        user_all_embeddings = U + torch.sparse.mm(self.sparse_norm_adj, V)
        item_all_embeddings = V
        return (user_all_embeddings, item_all_embeddings)

