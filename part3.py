'''
This file modifies the Vitebri algorithm to return the 5-th best path.
'''
import math
import copy
from part1 import *
from part2 import *
from utils import *


def viterbi_5best(transition_counts, emission_counts, observations, train_obs):
    neg_inf = (-math.inf)

    # Retrieving a list of all possible states
    states = list(transition_counts.keys())
    states.remove("START") # We don't need START and STOP states - will  be handled separately

    # List to store final paths of all observations
    ultimate_path = []
    for observation in observations: 
        # Initializing pi scores
        pi_scores = {}
        pi_scores[0] = {}
        # Sentence length
        n = len(observation)

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
                    pi_scores[0][u] = ("START",  log_transition_param+log_emission_param)

                    pi_scores_copy = copy.deepcopy(pi_scores)
                    

            # From the second word till the second last word
            else:
                # Creating a copy so we can find the 5 maximum scores
                pi_scores_copy[j+1] = {}
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

                        if j == 1 :
                            current_score = pi_scores[j-1][v][1] + log_transition_param + log_emission_param
                            score_list.append(current_score)
                        else:
                            currentscores = [[pi_scores[j-1][v][k][1] for k in range(5)][l] + log_transition_param + log_emission_param for l in range(5)] 
                            for score in currentscores:
                                score_list.append(score)

                    # The 5 highest Scores are stored in best5scores in the format: [highest, 2nd highest, ... , 5th highest]
                    # state_ans holds the best 5 states in decreasing order in the format [best state, 2nd best state, ..... , 5th best state]
                    best5scores = [] 
                    state_ans = []
                    score_list_copy = copy.deepcopy(score_list)
                    for num in range(5):
                        best5scores.append(max(score_list_copy))
                        state_ans.append(states[score_list_copy.index(best5scores[num]) // 5])  
                        score_list_copy[score_list.index(best5scores[num])] = neg_inf
                
                    pi_scores[j][u] = tuple((state_ans[m], best5scores[m]) for m in range(5))
                    
        # Transitioning from last word to STOP
        pi_scores[n] = {}
        pi_scores_copy[n] = {}
        final_scores = []
        for u in states:
            transition_param = get_transition_parameters(transition_counts, 'START', u)
            if transition_param == 0:
                log_transition_param = neg_inf
            else:
                log_transition_param = math.log(transition_param)

            if(type(pi_scores[n-1][u][0])==tuple):
                current_score = [[pi_scores[n-1][u][p][1] for p in range(5)][l] + log_transition_param + log_emission_param for l in range(5)]
            else:
                a=pi_scores[n-1][u]
                current_score = [a[1]+ log_transition_param + log_emission_param]    

            for score in current_score:
                final_scores.append(score)
                    
        # Fnd 5 highest scores in final_scores
        # stop_scores stores the 5 highest scores in the format: [highest, 2nd highest, 3rd highest, 4th highest , 5th highest]
        # state_ans stores the best 5 states in the format: [highest, 2nd highest, 3rd highest, 4th highest , 5th highest]
        stop_scores = []
        state_ans = []
        final_scores_copy = copy.deepcopy(final_scores)
        for i in range(5):
            stop_scores.append(max(final_scores_copy))
            state_ans.append(states[final_scores.index(stop_scores[i]) // 5])
            final_scores_copy[final_scores.index(stop_scores[i])] = neg_inf
        pi_scores[n][u] = tuple((state_ans[m], stop_scores[m]) for m in range(5))
                 
        # Backward algorithm
        # best_5 stores the 5 best paths as [best path, 2nd best path, ... , 5th best path]
        # last_states tracks the last state of each of the 5 best paths in the format [best path's last, 2nd best path's last, ... , 5th best path's last]
        best_5 = []
        last_states = [] 
        for i in range(5):
            path = ["STOP"]
            last = list(pi_scores[n].values())[0][i][0]
            last_states.append(last)
            path.insert(0, last)
            best_5.append(path) # Excluding START
        
        for i in range(5):
            for k in range(n-1, 0, -1):
                if k == 0:
                    last = pi_scores[k][best_5[i][0]][0] 
                else:
                    last = pi_scores[k][best_5[i][0]][0][0]
                best_5[i].insert(0, last)
        
        ultimate_path.append(best_5[4])
    return ultimate_path


if __name__=="__main__":
    dataset = input("Please enter dataset ('ES' or 'RU'): ")
    if dataset == "ES":
        test_data_path= f'./ES/dev.in'
        test_output_path=f'./ES/dev.p4.out'
        train_path='./ES/train'
    elif dataset == "RU":
        test_data_path=f'./RU/dev.in'
        test_output_path=f'./RU/dev.p4.out'
        train_path='./RU/train'
    
    train_obs, emission_counts = emission_counting(train_path)
    transition_counts = transition_counting(train_path)
    sequence = data_dump(test_data_path)
    viterbi_outputs = viterbi_5best(transition_counts,emission_counts, sequence, train_obs)
    #print(viterbi_outputs)

    with open(test_data_path, "r", encoding="utf8") as f:
                lines = f.readlines()

    with open(test_output_path, "w", encoding="utf8") as g:
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

    
