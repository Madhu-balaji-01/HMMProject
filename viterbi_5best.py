import math
import copy
from part1_final import *
from part2 import *
from utils import *

# def viterbi_5_best(emission_counts, transition_counts, observations, train_obs):
#     # Retrieving a list of all possible states
#     states = list(transition_counts.keys())
#     states.remove('START') # We don't need START and STOP states - will  be handled separately

#     neg_inf = (-math.inf)
    
#     ultimate_path = []
#     # 'observations' are of the form [['sentence1_word1', 'sentence1_word2', 'sentence1_word3'], ['sentence2_word1']]
#     for observation in observations: 
#         # Initializing pi scores
#         pi_scores = {}
#         pi_scores[0] = {'START' : 1}
#         # Sentence length
#         n = len(observation)
        
#         path_list = []
#         for j in range(n): 
#             pi_scores[j+1] = {}

#             # If an unseen word occurs, replace with '#UNK#' token
#             if observation[j] not in train_obs:
#                 x = "#UNK#"
#             else:
#                 x = observation[j]
            
#             # Transition from 'START' to first state and emission of first state to first word 
#             if j==0:
#                 for u in states:
#                     # To avoid underflow issue, we take log of emission and transition params
#                     transition_param = get_transition_parameters(transition_counts, 'START', u)
#                     if transition_param == 0:
#                         log_transition_param = neg_inf
#                     else:
#                         log_transition_param = math.log(transition_param)

#                     emission_param = estimate_emission_param(emission_counts, x, u)
#                     if emission_param == 0:
#                         log_emission_param = neg_inf
#                     else:
#                         log_emission_param = math.log(emission_param)

#                     # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
#                     # pi_scores[0]['START'] = 1 and log(1) = 0 so we ignore this term
#                     current_score = log_emission_param + log_transition_param
#                     # Now we store all state names as well as the score, not only the max 
#                     pi_scores[j+1][u] = {"START": current_score}

#                 # Creating a copy of pi_scores 
#                 copy_of_pi_scores = copy.deepcopy(pi_scores)
               
#             # From the second word till the second last word
#             else:
#                 for u in states:
#                     # To avoid underflow issue, we take log of emission and transition params
#                     emission_param = estimate_emission_param(emission_counts, x, u)
#                     if emission_param == 0:
#                         log_emission_param = neg_inf
#                     else:
#                         log_emission_param = math.log(emission_param)

#                     score_list = []
#                     score_list_copy = []
#                     for v in states:
#                         transition_param =  get_transition_parameters(transition_counts, v, u)
#                         if transition_param == 0:
#                             log_transition_param = neg_inf
#                         else:
#                             log_transition_param = math.log(transition_param)
                    
#                         # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
#                         current_score = pi_scores[j][v] + log_emission_param + log_transition_param
#                         score_list.append(current_score)

#                     # Storing the maximum score over all v's
#                     max_score = max(score_list)
#                     pi_scores[j+1][u] = max_score

#                 # # Storing the v that gives maximum score
#                 # path_list.append(max(score_dict, key = score_dict.get))

                
#         # Transition from last state to 'STOP'
#         pi_scores[n+1] = {}
#         for u in states:
#             transition_param = get_transition_parameters(transition_counts, u, 'STOP')
#             if transition_param == 0:
#                 log_transition_param = neg_inf
#             else:
#                 log_transition_param = math.log(transition_param)
             
#             pi_scores[n+1][u] = pi_scores[n][u] + log_transition_param  # No emission for STOP state
        
#         # Backward algorithm
#         back_tracker = []
        
#         for u_star in range(n,0,-1):
#             temp = max(pi_scores[u_star], key=pi_scores[u_star].get)
#             back_tracker.insert(0,temp)
            
#         ultimate_path.append(back_tracker) 
#         # print('Ultimate', ultimate_path)
    
#     return(ultimate_path)    

# train_obs, emission_counts = emission_counting('train')
# transition_counts = transition_counting('train')
# observations = data_dump('./ES/dev.in')
# viterbi_outputs = viterbi(emission_counts, transition_counts,  observations, train_obs)
# # print(viterbi_outputs)
# # all_prediction = l

# with open('./ES/dev.in', "r", encoding="utf8") as f:
#             lines = f.readlines()
            
# with open('./ES/dev.p2.out', "w", encoding="utf8") as g:
#     k = 0
#     num_lines = 0
#     for j in range(len(lines)):
#         word = lines[j].strip()
        
#         if (word != ""):
#             print('1',j - num_lines)
#             path = viterbi_outputs[k][j - num_lines]
#             g.write(word + " " + path)
#             g.write("\n")

#         else:
#             k+=1
#             num_lines = j + 1
#             g.write("\n")

def viterbi_5(emission_counts, transition_counts, sequence, states):
    
    temp = 1
    neg_inf = (-math.inf)
    
    pi_scores = {}
    pi_scores[0] = {"START": [[1,[]]]}
    
        
    for obs in sequence:
        pi_scores[temp] = {}
        
        if obs not in train_obs:
            obs = "#UNK#"
            
        for state in states:
            params_dict = {}
            
            for state_prev in pi_scores[temp-1].keys():
                
                emission_param = estimate_emission_param(emission_counts, obs, state)
                if emission_param == 0:
                    log_emission_param = neg_inf
                else:
                    log_emission_param = math.log(emission_param)
                    
                transition_param = get_transition_parameters(transition_counts, state_prev, state)
                if transition_param == 0:
                    log_transition_param = neg_inf
                else:
                    log_transition_param = math.log(transition_param)
                    
                if (log_transition_param != neg_inf) and (log_transition_param != neg_inf):
                    
                    for i in range(len(pi_scores[temp-1])):
                        params_dict[pi_scores[temp-1][state_prev][i][1]+[state_prev]] = pi_scores[temp-1][state_prev][i][0] + log_transition_param + log_emission_param
                        
                else:
                    params_dict[pi_scores[temp-1][state_prev][i][1]+[state_prev]] = neg_inf
                    
                # Key - 0, Values - 1
            all_probs = [0, sorted(params_dict.values(), reverse = True)]
            all_probs[0] = sorted(params_dict, key=params_dict.get, reverse=True)
                
            if len(all_probs[1] == 1):
                pi_scores[temp][state] = [[all_probs[1][0], list(all_probs[0][0])]]
                    
            else:
                pi_scores[temp][state] = [[all_probs[1][0], list(all_probs[0][0])], [all_probs[1][1], list(all_probs[0][1])], [all_probs[1][2], list(all_probs[0][2])]]
                    
            
        temp = temp + 1
                
            
    params_dict = {}
    for state_prev in pi_scores[temp-1].keys():
        transition_param = get_transition_parameters(transition_counts, state_prev, "STOP")
        if transition_param == 0:
            log_transition_param = neg_inf
        else:
            log_transition_param = math.log(transition_param)
                
        if log_transition_params != neg_inf:
            for i in range(pi_scores[temp-1][state_prev]):
                params_dict[pi_scores[temp-1][state_prev][i][1] + [state_prev]] = pi_scores[temp-1][state_prev][i][0] + log_transition_param
                    
        else:
            for i in range(pi_scores[temp-1][state_prev]):
                params_dict[pi_scores[temp-1][state_prev][i][1] + [state_prev]] = neg_inf
                    
    all_probs = [0, sorted(params_dict.values(), reverse = True)]
    all_probs[0] = sorted(params_dict, key=params_dict.get, reverse=True)
            
    viterbi5 = all_probs[0][2]
            
    prediction = []
    for ind, observation in enumerate(sequence):
        prediction.append([observation, viterbi5[ind+1]])
                
            
    return prediction


train_obs, emission_counts = emission_counting('./ES/train')
transition_counts = transition_counting('./ES/train')
sequence = data_dump('./ES/dev.in')
viterbi_outputs = viterbi5(emission_counts, transition_counts,  sequence, train_obs)
            
