import math
from part1 import *
from part2 import *
from utils import *

def viterbi(emission_counts, transition_counts, observations, train_obs):
    # Retrieving a list of all possible states
    states = list(transition_counts.keys())
    states.remove('START') # We don't need START and STOP states - will  be handled separately

    neg_inf = (-math.inf)
    
    ultimate = []
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

                    score_dict = {}
                    for v in states:
                        transition_param =  get_transition_parameters(transition_counts, v, u)
                        if transition_param == 0:
                            log_transition_param = neg_inf
                        else:
                            log_transition_param = math.log(transition_param)

                        # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
                        score_dict[v] = pi_scores[j][v] + log_emission_param + log_transition_param
                        #score_list.append(pi_scores[j][v] + log_emission_param + log_transition_param)

                    # Storing the maximum score over all v's
                    max_score = max(list(score_dict.values()))
                    pi_scores[j+1][u] = max_score

                    # Storing the v that gives maximum score
                    path_list.append(max(score_dict, key = score_dict.get))

                
        # Transition from last state to 'STOP'
        pi_scores[n+1] = {}
        score_dict ={}
        for u in states:
            transition_param = get_transition_parameters(transition_counts, u, 'STOP')
            if transition_param == 0:
                log_transition_param = neg_inf
            else:
                log_transition_param = math.log(transition_param)

            pi_scores[n+1][u] = pi_scores[n][u] + log_transition_param  # No emission for STOP state
            
        path_list.append(max(pi_scores[n+1], key = pi_scores[n+1].get))
        print('Path', path_list)
        # max_score = max(score_list)
        # break
        
#         back_tracker = ["STOP"]
        
#         #temp = max(pi_scores[n], key=pi_scores[n].get)
        
#         #back_tracker.insert(0,temp)
        
#         for why_am_i_doing_this_i_want_to_go_back_to_blockchain in range(n,-1,-1):
            
#             temp = max(pi_scores[why_am_i_doing_this_i_want_to_go_back_to_blockchain], key=pi_scores[why_am_i_doing_this_i_want_to_go_back_to_blockchain].get)
#             back_tracker.insert(0,temp)
            
#         #print(back_tracker)
#         ultimate.append(back_tracker)
#         break   
#     return(ultimate)    

#         # pi_scores[j+1][u] = max_score
# #         path_list.append(states[score_list.index(max_score)])
# #         print(path_list)       

# train_obs, emission_counts = emission_counting('train')
# transition_counts = transition_counting('train')
# observations = data_dump('./ES/dev.in')
# l = viterbi(emission_counts, transition_counts,  observations, train_obs)

# all_prediction = l

# with open('./ES/dev.in', "r", encoding="utf8") as f:
#             lines = f.readlines()

# with open('./ES/dev.p2.out', "w", encoding="utf8") as g:
#     for j in range(len(lines)):
#         word = lines[j].strip()
#         if word != "\n":
#             tag = all_prediction[j]
#             if(tag != "\n"):
#                 g.write(word + " " + tag)
#                 g.write("\n")
#             else:
#                 g.write("\n")
                
#all_prediction = final_answers_part1('RU')
                
# with open('./RU/dev.in', "r", encoding="utf8") as f:
#             lines = f.readlines()

# with open('./RU/dev.p2.out', "w", encoding="utf8") as g:
#     for j in range(len(lines)):
#         word = lines[j].strip()
#         if word != "\n":
#             tag = all_prediction[j]
#             if(tag != "\n"):
#                 g.write(word + " " + tag)
#                 g.write("\n")
#             else:
#                 g.write("\n")s