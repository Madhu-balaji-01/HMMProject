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

                    # ans is a list that holds the N highest scores
                    # eg. ans = [highest, 2nd highest, 3rd highest, .... , N highest]
                    # state_ans is a list that holds the N best states, with decrementing "bests state" from left to right
                    # eg. state_ans = [best state, 2nd best state, ..... , N best state]

                    ans = [] 
                    state_ans = []
                    score_list_copy = copy.deepcopy(score_list)
                    for num in range(5):
                        ans.append(max(score_list_copy))
                        state_ans.append(states[score_list_copy.index(ans[num]) // 5])
                        score_list_copy[score_list.index(ans[num])] = neg_inf
                
                    # store nested tuple of N best states ((best state, score),(2nd best state, score),(3rd best state, score))
                    pi_scores[j][u] = tuple((state_ans[m], ans[m]) for m in range(5))
                    
        #print('scores',scores)
        # STOP STATE
        pi_scores[n] = {}
        pi_scores_copy[n] = {}
        stopmax = []

        for u in states:
            # Transition Probability
            transition_param = get_transition_parameters(transition_counts, 'START', u)
            if transition_param == 0:
                log_transition_param = neg_inf
            else:
                log_transition_param = math.log(transition_param)

            if(type(pi_scores[n-1][u][0])==tuple):
                stopscore = [[pi_scores[n-1][u][p][1] for p in range(5)][l] + log_transition_param + log_emission_param for l in range(5)]
            else:
                t=pi_scores[n-1][u]
                stopscore = [t[1]+ log_transition_param + log_emission_param]    

            for score in stopscore:
                stopmax.append(score)
                    
        # ARGMAX
        # code to find n highest scores in stopmax
        # stop is a list that holds the N highest scores to stop
        # eg. stop = [highest, 2nd highest, 3rd highest, ... , N highest]
        # state_ans is a list that holds the best N bests state
        # state_ans = [best state, 2nd best state, ... , N , best state]
        stop = []
        state_ans = []
        stopmax_copy = copy.deepcopy(stopmax)
        for i in range(5):
            stop.append(max(stopmax_copy))
            state_ans.append(states[stopmax.index(stop[i]) // 5])
            stopmax_copy[stopmax.index(stop[i])] = neg_inf
        pi_scores[n][u] = tuple((state_ans[m], stop[m]) for m in range(5))
                 
        # Backtracking path
        # N_bestPaths is list of N lists, that holds N best paths in decreasing order.
        # eg. N_bestPaths = [best path, 2nd best path, ... , N best path]
        # lasts is a list to trac k the last state of each of the N best paths. 
        # eg. lasts = [best path's last, 2nd best path's last, ... , N best path's last]
        best_5 = []
        lasts = [] 
        for i in range(5):
            path = ["STOP"]
            last = list(pi_scores[n].values())[0][i][0]
            lasts.append(last)
            path.insert(0, last)
            # print(path)
            best_5.append(path) # Excluding START
        
        for i in range(5):
            for k in range(n-1, 0, -1):
                if k == 0:
                    last = pi_scores[k][best_5[i][0]][0] 
                else:
                    last = pi_scores[k][best_5[i][0]][0][0]
                best_5[i].insert(0, last)
        #break
        ultimate_path.append(best_5[4])
    return ultimate_path


train_obs, emission_counts = emission_counting('./RU/train')
transition_counts = transition_counting('./RU/train')
sequence = data_dump('./RU/dev.in')
# sequences = []

# for i in sequence:
#     for j in i: 
#         sequences.append(j)
# print(sequences)

viterbi_outputs = viterbi_5best(transition_counts,emission_counts, sequence, train_obs)
#print(viterbi_outputs)

with open('./RU/dev.in', "r", encoding="utf8") as f:
            lines = f.readlines()

with open('./RU/dev.p3_test.out', "w", encoding="utf8") as g:
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



# if __name__ == '__main__':
#     part4_dataset = ["RU"]

#     for i in part4_dataset:
#         """
#         train = "Data/{folder}/train".format(folder = i)
#         evaluation = "Data/{folder}/dev.in".format(folder = i)
#         """

#         root_dir = './'
        
#         train = root_dir + "{folder}/train".format(folder = i)
#         evaluation = root_dir + "{folder}/dev.in".format(folder = i)
        
#         # training
#         transition_tracker = transition_counting(train)

#         obs_all, emission_tracker = emission_counting(train)
        
#         # evaluation
#         with open(evaluation, "r", encoding= 'utf8') as f:
#             # readlines() returns a list containing each line in the file as a list item
#             # each line is a word
#             lines = f.readlines()
        
#         # track each observations's prediction labels
#         # each word's prediction label will be an element of this list
#         observations = []
        
#         # list containing all prediction labels
#         # observationss are separated with element "\n" in between
#         all_prediction = []
#         #print(i)
#         # initialise N
#         N = 5
#         # each line is a word
#         for line in lines:        
#             if line != "\n":
#                 line = line.strip()
#                 observations.append(line)
#             else:
#                 observations_prediction = viterbi_5(N, emission_tracker, transition_tracker, obs_all, observations)
#                 observations_prediction.remove("START")
#                 observations_prediction.remove("STOP")
#                 all_prediction = all_prediction + observations_prediction
#                 all_prediction = all_prediction + ["\n"]
#                 observations = []
        
#         assert len(lines) == len(all_prediction)
#         # create output file
#         with open('./' + "{folder}/dev.p3.out".format(folder = i), "w", encoding='utf8') as g:
#             for j in range(len(lines)):
#                 word = lines[j].strip()
#                 if word != "\n":
#                     tag = all_prediction[j]
#                     if(tag != "\n"):
#                         g.write(word + " " + tag)
#                         g.write("\n")
#                     else:
#                         g.write("\n")

#     print("done")