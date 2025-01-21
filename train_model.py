from SELFRec import SELFRec
import yaml
import time
import argparse
import torch
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--model', type=str, default='THRGB_MHCN')
    parser.add_argument('--strategy', type=str, default='homo_first')
    parser.add_argument('--zeta', type=float, default=0.7)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    opts = parser.parse_args()
    dataset = opts.dataset# lastfm or douban or yelp

    s = time.time()

    with open('conf/'+opts.model+'.yaml', "r") as file: # SHaRe_MHCN.yaml or diffnet.yaml or LightGCN.yaml
        conf = yaml.load(file, Loader=yaml.FullLoader)

    conf['data'] = dataset
    conf['zeta'] =opts.zeta
    conf['lambda'] = opts.lamb
    conf['n_layer'] =opts.n_layer
    conf['gpu']=opts.gpu
    conf['strategy']=opts.strategy
    rec = SELFRec(conf)
    rec.execute()

    e = time.time()
    print("Running time: %f s" % (e - s))
