import math
from part1 import *
from part2 import *
from utils import *

def viterbi(emission_counts, transition_counts, observations):
    # Retrieving a list of all possible states
    states = list(transition_counts.keys())
    states.append('STOP') # START is already in the list, no need to append

    # 'observations' are of the form [['sentence1_word1', 'sentence1_word2', 'sentence1_word3'], ['sentence2_word1']]
    for observation in observations: 
        # Initializing pi scores
        pi_scores = {}
        pi_scores[0] = {'START' : 1}
        # Sentence length
        n = len(observation)
        path_list = []

        for j in range(n): 
            pi_scores[j+1] = {}
            if j==0:
                score_list = []
                for u in states:
                    print(u)
                    emission_param = estimate_emission_param(emission_counts, observation[j+1], u)
                    # emission_param = 0.3
                    # To avoid underflow issue, we take log of emission and transition params
                    log_emission_param = math.log(emission_param)
                    transition_param = get_transition_params(transition_counts, 'START', u)
                    # transition_param = 0.8
                    log_transition_param = math.log(transition_param)
                    # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
                    # pi_scores[0]['START'] = 1 and log(1) = 0 so we ignore this term
                    score_list.append(log_emission_param + log_transition_param)

            else:
                for u in states:
                    # emission_param = get_emission_params(emission_counts, u, observation[j+1])
                    emission_param = 0.3
                    # To avoid underflow issue, we take log of emission and transition params
                    log_emission_param = math.log(emission_param)

                    score_list = []
                    for v in states:
                        # transition_param = get_transition_params(transition_counts, v, u)
                        transition_param = 0.8
                        log_transition_param = math.log(transition_param)
                        # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
                        score_list.append(pi_scores[j][v] + log_emission_param + log_transition_param)

                    max_score = max(score_list)
                    pi_scores[j+1][u] = max_score
                path_list.append(states[score_list.index(max_score)])
                print(path_list)
        
        # STOP state
        pi_scores[n] = {}
        for u in states:
            transition_param = get_transition_params(transition_counts, u, 'STOP')
            # transition_param = 0.8
            log_transition_param = math.log(transition_param)
            pi_scores[n][u] = pi_scores[n-1][u] + log_transition_param

        path_list.append(states[score_list.index(max_score)])



            
obs, emission_counts = emission_counting('train')
transition_counts = transition_counting('train')
observations = data_dump('./ES/dev.in')

viterbi(emission_counts, transition_counts,  observations)
