'''
generate a caption
something wrong
'''

import pickle as pkl
from scipy.io import loadmat
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import capgen
import gen_sample_2 as gs
import numpy

def load_pkl(path):
    print 'loading', path
    pkl_file = None
    with open(path, 'rb') as f: pkl_file = pkl.load(f)
    f.close()
    # print 'pkl loaded'
    return pkl_file

def load_model(model_path):
    # print 'loading model'
    model = model_path
    options = load_pkl(model + '.pkl')
    # build the sampling functions and model
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.), name='use_noise')
    params = capgen.init_params(options)
    params = capgen.load_params(model, params)
    tparams = capgen.init_tparams(params)
    f_init, f_next = capgen.build_sampler(tparams, options, use_noise, trng)
    trng, use_noise, inps, alphas, alphas_samples, cost, opt_outs = capgen.build_model(tparams, options)
    # print 'done'
    return tparams, f_init, f_next, options, trng

def re_gen(y):
    # reload(gs)
    gs.tparams, gs.f_init, gs.f_next, gs.options = [tparams, f_init, f_next, options]
    sample, score = gs.gen_sample(y)
    return sample[0], score

def example_code():
    cap_path='data/flickr8k/flicker_8k_cap.test.pkl'
    feat_path = 'data/flickr8k/196x512_perfile_dense_name_resizecrop/'
    test_cap = pkl.load(open(cap_path, 'rb'))

    feat_list = []
    _y = loadmat(feat_path + str(test_cap[0][1]) + '.mat')['feats']
    _y = numpy.array(_y).reshape([14*14, 512]).astype('float32')
    feat_list.append(_y)
    sample, score = re_gen(feat_list[0])
    while sample[-1] == 0:
        sample = sample[:-1]
    print ' '.join(map(lambda w: word_idict[w] if w in word_idict else '<UNK>', sample))

def calc_flickr8k_BLUE():
    import bleu
    cap_path = 'data/flickr8k/flicker_8k_cap.test.pkl'
    feat_path = 'data/flickr8k/196x512_perfile_dense_name_resizecrop/'

    test_cap = pkl.load(open(cap_path, 'rb'))
    r_c = {} # key:image file name, value:captions list
    for i in test_cap:
        if i[1] in r_c:
            r_c[i[1]].append(i[0])
        else:
            r_c[i[1]] =[i[0]]
    rc_k = r_c.keys()

    candidate, references = [], []
    print 'generating samples'
    for k in rc_k:
        _y = loadmat(feat_path + str(k) + '.mat')['feats']
        _y = numpy.array(_y).reshape([14*14, 512]).astype('float32')
        sample_i, _ = re_gen(_y)
        while sample_i[-1] == 0:
            sample_i = sample_i[:-1]
        sample_w =  ' '.join(map(lambda w: word_idict[w] if w in word_idict else '<UNK>', sample_i))
        candidate.append(sample_w)
        references.append(r_c[k])

    bleu.calc_BLEU(references, candidate)

tparams, f_init, f_next, options, trng = [], [], [], [], []
from flickr8k import load_data
_, _, _, worddict = load_data(load_train=False, load_dev=False, load_test=False)

word_idict = dict()
for kk, vv in worddict.iteritems():
    word_idict[vv] = kk
word_idict[0] = '<eos>'
word_idict[1] = 'UNK'
# calc_flickr8k_BLUE()

import os
path = 'cv/normal/'
pkl_name = 'flickr8k.npz.pkl'
for i in range(1, 100):
    model_name = 'flickr8k.npz_epoch_' + str(i) + '.npz'
    new_pkl_name = model_name + '.pkl'
    os.rename(path + pkl_name, path + new_pkl_name)
    pkl_name = new_pkl_name
    
    tparams, f_init, f_next, options, trng = load_model(path + model_name)
    calc_flickr8k_BLUE()


