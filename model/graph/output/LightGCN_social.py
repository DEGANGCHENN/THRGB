import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
from data.social import Relation
import pickle

class LightGCN_social(GraphRecommender):
    
    def __init__(self, conf, training_set, valid_set, test_set, **kwargs):
        super(LightGCN_social, self).__init__(conf, training_set, valid_set, test_set)
        self.n_layers = self.config['n_layer']
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        S = self.social_data.get_social_mat()
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers, S)

    
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



class LGCN_Encoder(nn.Module):
    
    def __init__(self, data, emb_size, n_layers, sparse_S):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.sparse_S = TorchGraphInterface.convert_sparse_mat_to_tensor(sparse_S).cuda()

    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))) })
        return embedding_dict

    
    def forward(self):
        ego_embeddings = torch.cat([ self.embedding_dict['user_emb'],self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            U, V = torch.split(ego_embeddings, [self.data.user_num,self.data.item_num])
            U_s = torch.sparse.mm(self.sparse_S, U)
            U = U + U_s
            ego_embeddings = torch.cat([U,V], dim = 0)
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim = 1)
        all_embeddings = torch.mean(all_embeddings, dim = 1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings

