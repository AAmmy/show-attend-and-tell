import numpy
import copy

tparams, f_init, f_next, options = [None, None, None, None]
trng = None
k = 5
maxlen = 30
stochastic = False

def gen_sample(ctx0, w_idx = []):
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    hyp_memories = []

    # only matters if we use lstm encoder
    rval = f_init(ctx0)
    ctx0 = rval[0]
    next_state = []
    next_memory = []
    # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(rval[1+lidx])
        next_state[-1] = next_state[-1].reshape([1, next_state[-1].shape[0]])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[1+options['n_layers_lstm']+lidx])
        next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].shape[0]])
    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)  
    next_w = -1 * numpy.ones((1,)).astype('int64')

    genSampleFlag = True
    if len(w_idx) > 0 and w_idx[-1] == -1:
        genSampleFlag = False
        w_idx[-1] = 0

    pre_cost = 0
    for ii in xrange(maxlen):
        if not genSampleFlag:
            if ii >= len(w_idx):
                # ipdb.set_trace()
                return hyp_samples, hyp_scores

        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = f_next(*([next_w, ctx0] + next_state + next_memory))
        next_p = rval[0]
        next_w = rval[1]

        # extract all the states and memories
        next_state = []
        next_memory = []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[2+lidx])
            next_memory.append(rval[2+options['n_layers_lstm']+lidx])

        if stochastic:
            sample.append(next_w[0]) # if we are using stochastic sampling this easy
            sample_score += next_p[0,next_w[0]]
            if next_w[0] == 0:
                break
        else:
            cand_scores = hyp_scores[:,None] - numpy.log(next_p) 
            cand_flat = cand_scores.flatten()
            # ranks_flat = cand_flat.argsort()[:(k-dead_k)] # (k-dead_k) numpy array of with min nll
            # ridx = numpy.random.randint(0, 4)
            # ranks_flat = cand_flat.argsort()[ridx:ridx + 1]
            if not genSampleFlag:
                # ranks_flat = cand_flat.argsort()[:(k-dead_k)] # (k-dead_k) numpy array of with min nll
                ridx = w_idx[ii]
                ranks_flat = numpy.array([ridx])
            else:
                if w_idx == []:
                    ranks_flat = cand_flat.argsort()[:(k-dead_k)]
                else:
                    if not w_idx[-1] == -1:
                        if len(w_idx) <= ii:
                            ridx = 0
                        else:
                            ridx = w_idx[ii]
                    else:
                        ridx = w_idx[ii]
                    ranks_flat = numpy.array([cand_flat.argsort()[ridx]])
            # print ridx,
            
            # print '{0:2d}'.format(ii),
            # print '{0:5d}'.format(ranks_flat[0]), ':',
            
            # print ii
            # print cand_flat.argsort()[ridx]
            # print cand_flat.argsort()[:ridx + 1]
            # print cand_flat.argsort()
            

            voc_size = next_p.shape[1]
            # indexing into the correct selected captions
            trans_indices = ranks_flat / voc_size
            # word_indices = ranks_flat % voc_size

            word_indices = ranks_flat % voc_size
            # print ''
            # print ranks_flat, voc_size, trans_indices, word_indices
            """
            if genSampleFlag:
                word_indices = ranks_flat % voc_size
            else:
                word_indices = numpy.array([w_idx[ii]])
            """

            costs = cand_flat[ranks_flat] # extract costs from top hypothesis
            
            now_cost = costs[0] - pre_cost
            pre_cost = costs[0]
            # print ('%03.4f' % costs[0]).rjust(7), ('%3.4f' % now_cost).rjust(7)

            # a bunch of lists to hold future hypothesis
            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_states.append([])
            new_hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_memories.append([])

            # get the corresponding hypothesis and append the predicted word
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx]) # copy in the cost of that hypothesis 
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))

            # check the finished samples for <eos> character
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_states.append([])
            hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_memories.append([])

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1 # completed sample!
                else:
                    new_live_k += 1 # collect collect correct states/memories
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_states[lidx].append(new_hyp_states[lidx][idx])
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
            hyp_scores = numpy.array(hyp_scores)
            # print hyp_samples
            # print hyp_scores
            # print hyp_states
            # print new_hyp_scores
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = []
            for lidx in xrange(options['n_layers_lstm']):
                next_state.append(numpy.array(hyp_states[lidx]))
            next_memory = []
            for lidx in xrange(options['n_layers_lstm']):
                next_memory.append(numpy.array(hyp_memories[lidx]))

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score

