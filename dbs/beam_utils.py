import torch
from torch import nn

def add_diversity(beam_seq_table,
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
            logprobsf[sub_beam]
            prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - opt._lambda
    return unaug_logprobsf


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
        pass

    for vix in range(beam_size):
        v = candidates[vix]
        v['kept'] = True
        # Fork beam index 'q' into index 'vix'
        if t > 1:
            #TODO: fork beam index 'q' into index 'vix'
            pass

        for state_ix in range(len(new_state)):
            new_state[state_ix][vix] = state[state_ix][v.q]

        

    