import math
from os import name
from part1 import *
from part2 import *
from utils import *

def viterbi(emission_counts, transition_counts, observations, train_obs):
    # Retrieving a list of all possible states
    states = list(transition_counts.keys())
    states.remove('START') # We don't need START and STOP states - will  be handled separately

    neg_inf = (-math.inf)
    
    ultimate_path = []
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

            # If an unseen word occurs, replace with '#UNK#' token
            if observation[j] not in train_obs:
                x = "#UNK#"
            else:
                x = observation[j]
            
            # Transition from 'START' to first state and emission of first state to first word 
            if j==0:
                for u in states:
                    # To avoid underflow issue, we take log of emission and transition params
                    transition_param = get_transition_parameters(transition_counts, 'START', u)
                    if transition_param == 0:
                        log_transition_param = neg_inf
                    else:
                        log_transition_param = math.log(transition_param)

                    emission_param = estimate_emission_param(emission_counts, x, u)
                    if emission_param == 0:
                        log_emission_param = neg_inf
                    else:
                        log_emission_param = math.log(emission_param)

                    # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
                    # pi_scores[0]['START'] = 1 and log(1) = 0 so we ignore this term
                    pi_scores[j+1][u] = log_emission_param + log_transition_param
               
            # From the second word till the second last word
            else:
                for u in states:
                    # To avoid underflow issue, we take log of emission and transition params
                    emission_param = estimate_emission_param(emission_counts, x, u)
                    if emission_param == 0:
                        log_emission_param = neg_inf
                    else:
                        log_emission_param = math.log(emission_param)

                    score_list = []
                    for v in states:
                        transition_param =  get_transition_parameters(transition_counts, v, u)
                        if transition_param == 0:
                            log_transition_param = neg_inf
                        else:
                            log_transition_param = math.log(transition_param)

                        # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
                        score_list.append(pi_scores[j][v] + log_emission_param + log_transition_param)

                    # Storing the maximum score over all v's
                    max_score = max(score_list)
                    pi_scores[j+1][u] = max_score

                # # Storing the v that gives maximum score
                # path_list.append(max(score_dict, key = score_dict.get))

                
        # Transition from last state to 'STOP'
        pi_scores[n+1] = {}
        for u in states:
            transition_param = get_transition_parameters(transition_counts, u, 'STOP')
            if transition_param == 0:
                log_transition_param = neg_inf
            else:
                log_transition_param = math.log(transition_param)
             
            pi_scores[n+1][u] = pi_scores[n][u] + log_transition_param  # No emission for STOP state
        
        # Backward algorithm
        back_tracker = []
        
        for u_star in range(n,0,-1):
            temp = max(pi_scores[u_star], key=pi_scores[u_star].get)
            back_tracker.insert(0,temp)
            
        ultimate_path.append(back_tracker) 
        print('Ultimate', ultimate_path)
    
    return (ultimate_path)    

if __name__=="__main__":
    train_obs, emission_counts = emission_counting('./RU/train')
    transition_counts = transition_counting('./RU/train')
    observations = data_dump('./RU/dev.in')
    viterbi_outputs = viterbi(emission_counts, transition_counts,  observations, train_obs)
    # print(viterbi_outputs)
    # all_prediction = l

    with open('./RU/dev.in', "r", encoding="utf8") as f:
                lines = f.readlines()
                
    with open('./RU/dev.p2.out', "w", encoding="utf8") as g:
        k = 0
        num_lines = 0
        for j in range(len(lines)):
            word = lines[j].strip()
            
            if (word != ""):
                path = viterbi_outputs[k][j - num_lines]
                g.write(word + " " + path)
                g.write("\n")

            else:
                k+=1
                num_lines = j + 1
                g.write("\n")
            