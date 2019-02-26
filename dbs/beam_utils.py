import torch
from torch import nn

def __add_diversity(beam_seq_table,
                  logprobsf,
                  t,
                  divm,
                  opt,
                  bdash): 
    """Computes the similarity score to be augmented
    """
    local_time = t - divm + 1
    unaug_logprobfs = logprobsf.clone()
    for prev_choice in range(divm-1):
        prev_decisions = beam_seq_table[prev_choice][local_time]
        for sub_beam in range(bdash):
            logprobsf[sub_beam][prev_decisions[prev_labels]] = \
                logprobsf[sub_beam][prev_decisions[prev_labels]] - opt['lambda']
    return unaug_logprobsf

def __clone_list(lst):
    # given a list of tensors, clones them all
    result = []
    for item in lst:
        result.append(item.clone())
    return result


def __beam_step(logprobsf,
                unaug_logprobsf,
                beam_size,
                t,
                beam_seq,
                beam_seq_logprobs,
                beam_logprobs_sum,
                state,
                stanford_lambda):
    """ Does one step of classical beam search

    Keyword arguments:
    logprobsf -- probabilities augmented after diversity
    unaug_logprobsf -- 
    beam size -- beam size
    t -- time instant
    beam_seq -- tensor containing the beams
    beam_seq_logprobs -- tensor containing joint logprobs

    Returns:
    beam_seq -- tensor containing the word indices of the decoded captions
    beam_seq_logprobs -- log-probabilities of each decision made, same size as beam_seq
    beam_logpros_sum -- joint log-probability of each beam
    """

    ys, index = torch.sort(logprobfs, 1, True)
    candidates = []
    cols = min(beam_size, ys.size()[1])
    rows = beam_size

    if t == 1:
        rows = 1

    for c in range(cols):
        for q in range(rows):
            local_logprob = ys[{q, c}]
            candidate_logprob = beam_logprobs_sum[q] + local_logprob
            # TODO local_unaug_logprob = unaug_logprobsf[[q, c]]
            local_unaug_logprob = unaug_logprobsf[q][index[q][c]]
            candidates.append({"c"=ix[q][c], "q"=q, "p"=candidate_logprob, "r"=local_unaug_logprob})

    candidates.sort(key=lambda x: x.p)
    new_state = state.copy()

    if t > 1:
        beam_seq_prev = beam_seq[0:t].clone()
        beam_seq_logprobs_prev = beam_seq_logprobs[0:t].clone()

    for vix in range(beam_size):
        v = candidates[vix]
        v['kept'] = True
        # Fork beam index 'q' into index 'vix'
        if t > 1:
            beam_seq[0:t][vix] = beam_seq_prev[:,v["q"]]
            beam_seq_logprobs[0:t][vix] = beam_seq_logprobs_prev[:,v['q']]

        #rearrange recurrent states
        for state_ix in range(len(new_state)):
            #copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v["q"]]

        #append new end terminal at the end of this beam
        beam_seq[t][vix] = v['c']
        beam_seq_logprobs[t][vix] = v['r']
        beam_logprobs_sum[vix] = v['p']

    state = new_state
    return (beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates)


def beam_search(init_params, opt):
    """ Implements beam search

    Calls __beam_step and returns the final set of beams
    Augments log-probabilities with the diversity terms when number of groups > 1
    
    #TODO: document params
    """

    # Initializing variables
    bdash = opt['B']/opt["M"]
    init_state = init_params[1]
    init_logprobs = init_params[2]
    state_table = []
    beam_seq_table = []
    beam_seq_logprobs_table = []
    beam_logprobs_sum_table = []
    to_stanford = opt['baseline']

    for i in range(opt['M']):
        state_table[i] = []
        beam_seq_table[i] = torch.zeros([opt['T'], bdash]).to(opt['device'])
        beam_seq_logprobs_table[i] = torch.zeros([opt['T'], bdash]).to(opt['device'])
        beam_logprobs_sum_table[i] = torch.zeros([bdash]).to(opt['device'])

    done_beams_table = [[] for i in range(opt['M'])]

    state = [torch.zeros([bdash, opt.rnn_size]).to(opt['device']) for h in range(opt['state_size'])]
    for h in range(opt['state_size']):
        for b in range(bdash):
            state[h][b] = init_state[h].clone()

    for i in range(opt['M']):
        logprobs_table[i] = torch.zeros([bdash, init_logprobs.size()[1]]).to(opt['device'])
        for j in range(bdash):
            logprobs_table[i][j] = init_logprobs.clone()

    # End initialization

    for t in range(opt['T']+opt['M']-1):
        vis_candidates = []
        for divm in range(opt['M']):
            if t >= divm and t <= opt['T']+divm-1:
                # add diversity
                logprobsf = logprobs_table[divm]

                # Suppress <UNK> tokens in the decoding
                logprobsf[:,logprobsf.size()[1]-1] = logprobsf[:,logprobsf.size()[1]-1] - 1000

                # Diversity is added here
                # The function directly modifies the logprobsf values and hence,
                # we need to return the unaugmented ones for sorting the candidates in the end
                # (mainly for historical reasons)

                unaug_logprobsf = __add_diversity(beam_seq_table, logprobsf, t, divm, opt, bdash)

                # Infer new beams
                beam_seq_table[divm],
                beam_seq_logprobs_table[divm],
                beam_logprobs_sum_table[divm],
                state_table[divm]
                candidates_divm = __beam_step(logprobsf,
                                              unaug_logprobsf,
                                              bdash,
                                              t - divm + 1,
                                              beam_seq_table[divm],
                                              beam_seq_logprobs_table[divm],
                                              beam_logprobs_sum_table[divm],
                                              state_table[divm],
                                              to_stanford)

                # if time is up, or end token is reach, then copy beams
                for vix in range(bdash):
                    is_first_end_token = (
                        (beam_seq_table[divm][:,vix}][t-divm+1] == opt['end_token']) and \
                        (torch.eq(beam_seq_table[divm][:,vix],opt['end_token']).sum() == 1)
                    )

                    final_time_without_end_token = (
                        (t == opt['T']+divm-1) and \
                        (torch.eq(beam_seq_table[divm][:,vix],opt['end_token']).sum() == 0)
                    )

                    if is_first_end_token or final_time_without_end_token:
                        final_beam = {
                            "seq": beam_seq_table[divm][:, vix].clone(),
                            "logps": beam_seq_logprobs_table[divm][:,vix].clone(),
                            "unaug_logp": beam_seq_logprobs_table[divm][:,vix].sum(),
                            "logp" = beam_logprobs_sum_table[divm][vix]
                        }
                        if export_vis:
                            final_beam['candidate'] = candidates_kept_for_divm[vix]
                        done_beams_table[divm].append(final_beam)

                    # don't continue beams from finished sequences
                    if is_first_end_token:
                        # make continuation of already completed sequences improbable
                        beam_logprobs_sum_table[divm][vix] = -1000

                it = beam_seq_table[divm][t-divm+1]
                out = opt['gen_logprobs'](it, state_table[divm])
                logprobs_table[divm] = out[len(out)].clone()
                temp_state = []
                for i in range(opt['state_size']):
                    temp_state.append(out[i])
                state_table[divm] = __clone_list(temp_state)
    
     # asdfkjasdf                   
    for i in range(opt["M"]):
         done_beams_table[i].sort(key=lambda x: a['logp'])
         done_beams_table[i] = done_beams_table[i][0:bdash]
    return done_beams_table
