from data.loader import FileIO
import torch
import random
import numpy as np

class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.training_data = FileIO.load_data_set(config['data_path']+config['data']+'/'+config['training.set'], config['model.type'])
        self.valid_data = FileIO.load_data_set(config['data_path']+config['data']+'/'+config['valid.set'], config['model.type'])
        self.test_data = FileIO.load_data_set(config['data_path']+config['data']+'/'+config['test.set'], config['model.type'])

        self.kwargs = {}
        if 'social.data' in config:
            if self.config['social_weight'] == True:
                social_data = FileIO.load_social_data(config['data_path']+config['data']+'/weighted_'+self.config['social.data'])
                self.kwargs['social.data'] = social_data
            else:
                social_data = FileIO.load_social_data(config['data_path']+config['data']+'/'+self.config['social.data'])
                self.kwargs['social.data'] = social_data
        # if config.contains('feature.data'):
        #     self.social_data = FileIO.loadFeature(config,self.config['feature.data'])
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.'+ self.config['model.type'] +'.output.' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)
        recommender = self.config['model.name'] + '(self.config,self.training_data, self.valid_data, self.test_data,**self.kwargs)'
        eval(recommender).execute()
