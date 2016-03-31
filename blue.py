'''
calculate BLEU score with perl script
    https://github.com/karpathy/neuraltalk/blob/master/eval/multi-bleu.perl
make list of references and candidate(prediction)
format:
all_references -> [[ref0_0, ref0_1, ... , ref0_4], [ref1_0, ref1_1, ... ref1_4], ...]
all_candidates -> [cand0, cand1, ...]

'''
import json
import os

def calc_BLEU(all_references, all_candidates):
    ref_num = 5
    # use perl script to eval BLEU score for fair comparison to other research work
    # first write intermediate files
    print 'writing intermediate files into eval/'
    open('eval/output', 'w').write('\n'.join(all_candidates))
    for q in xrange(ref_num):
        open('eval/reference'+`q`, 'w').write('\n'.join([x[q] for x in all_references]))
    # invoke the perl script to get BLEU scores
    print 'invoking eval/multi-bleu.perl script...'
    owd = os.getcwd()
    os.chdir('eval')
    os.system('./multi-bleu.perl reference < output')
    os.chdir(owd)

def example_BLEU(): # example code for BLEU, this shows 100% score
    path='data/flickr8k/'
    test_cap = pkl.load(open(path + 'flicker_8k_cap.test.pkl', 'rb'))
    r_c = {} # key:image file name, value:captions list
    for i in test_cap:
        if i[1] in r_c:
            r_c[i[1]].append(i[0])
        else:
            r_c[i[1]] =[i[0]]
    rc_k = r_c.keys()
    r_c_l = []
    for k in rc_k:
        r_c_l.append(r_c[k])
    all_references = r_c_l
    all_candidates = [x[0] for x in r_c_l] # a sentence in a reference
    calc_BLEU(all_references, all_candidates)
    print('the scores are 100.0/100.0/100.0/100.0 ?')

'''
all_references = [['a dog', 'dogs', 'there is a dog', 'there are dogs', 'dog!'],
                    ['a cat', 'cats', 'there is a cat', 'there are cats', 'cat!']]
all_candidates = ['dog', 'cat']
calc_BLEU(all_references, all_candidates) # BLEU = 100.0/0.0/0.0/0.0
'''
