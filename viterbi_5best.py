import math
import copy
from part1_final import *
from part2 import *
from utils import *

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
            
