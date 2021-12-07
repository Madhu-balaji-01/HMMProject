import math
from boobs import *
from part2 import *
from utils import *

def viterbi(emission_counts, transition_counts, observations, train_obs):
    # Retrieving a list of all possible states
    states = list(transition_counts.keys())
    states.remove('START')
    # states.append('STOP') # START is already in the list, no need to append

    neg_inf = (-math.inf)
    all_paths = []
    # 'observations' are of the form [['sentence1_word1', 'sentence1_word2', 'sentence1_word3'], ['sentence2_word1']]
    for observation in observations: 
        # Initializing pi scores
        pi_scores = {}
        pi_scores[0] = {'START' : 1}
        # Sentence length
        n = len(observation)
        
        path_list = []
        for j in range(n): 
            # print(j)
            
            pi_scores[j+1] = {}
            if observation[j] not in train_obs:
                x = "#UNK#"
            else:
                x = observation[j]
            # print(x)
            if j==0:
                score_list = []
                for u in states:
                    emission_param = estimate_emission_param(emission_counts, x, u)
                    # emission_param = 0.3
                    # To avoid underflow issue, we take log of emission and transition params
                    if emission_param == 0:
                        log_emission_param = neg_inf
                        # print("emisino",u)
                    else:
                        log_emission_param = math.log(emission_param)
        
                    transition_param = get_transition_parameters(transition_counts, 'START', u)
                    if transition_param == 0:
                        log_transition_param = neg_inf
                        # print("trans",u)
                    else:
                        log_transition_param = math.log(transition_param)

                    # transition_param = 0.8
                    # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
                    # pi_scores[0]['START'] = 1 and log(1) = 0 so we ignore this term
                    pi_scores[j+1][u] = log_emission_param + log_transition_param
                    # print(j+1,u,log_emission_param + log_transition_param)
                    # print(pi_scores)
                    score_list.append(log_emission_param + log_transition_param)
                    print(score_list)
                max_score = max(score_list)
                path_list.append(states[score_list.index(max_score)])
                

            else:
                for u in states:
                    emission_param = estimate_emission_param(emission_counts, x, u)
                    # print('emission', emission_param, u)
                    # emission_param = 0.3
                    # To avoid underflow issue, we take log of emission and transition params
                    if emission_param == 0:
                        log_emission_param = neg_inf
                    else:
                        log_emission_param = math.log(emission_param)

                    score_list = []
                    for v in states:
                        # print(v)
                        # print('pi_scores',pi_scores)
                        transition_param =  get_transition_parameters(transition_counts, v, u)
                        # print('trans', transition_param, v)
                        if transition_param == 0:
                            log_transition_param = neg_inf
                        else:
                            log_transition_param = math.log(transition_param)
                        # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
                        # print(j)
                        score_list.append(pi_scores[j][v] + log_emission_param + log_transition_param)
                        print(path_list)
                    max_score = max(score_list)
                    pi_scores[j+1][u] = max_score
                    print(j+1,u,max_score)
                
                print("max",max_score)
                path_list.append(states[score_list.index(max_score)])
                print(path_list)
        # print(pi_scores)
        
        # STOP state
        pi_scores[n+1] = {}
        score_list = []
        # print(pi_scores)
        for u in states:
            # print(pi_scores)
            transition_param = get_transition_parameters(transition_counts, u, 'STOP')
            # transition_param = 0.8
            if transition_param == 0:
                log_transition_param = neg_inf
            else:
                log_transition_param = math.log(transition_param)
            pi_scores[n+1][u] = pi_scores[n][u] + log_transition_param
            score_list.append(pi_scores[n+1][u])

        max_score = max(score_list)

        # pi_scores[j+1][u] = max_score
        path_list.append(states[score_list.index(max_score)])
        print(path_list)
        break
            
train_obs, emission_counts = emission_counting('train')
transition_counts = transition_counting('train')
observations = data_dump('./ES/dev.in')

# print ("comida" in train_obs)
# print(emission_counts)
viterbi(emission_counts, transition_counts,  observations, train_obs)
