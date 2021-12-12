import math
import copy
from part1_final import *
from part2 import *
from utils import *

def viterbi_5(N, emission_dict, transition_dict, observations, sentence):
    n = len(sentence)
    smallest = -9999999

    # Set of states excluding START
    states = list(transition_dict.keys())
    states.remove("START")

    """"initialize score dict
      scores = { position: {
        state_v: {
          (state_u1, score),    
          (state_u2, score),  
          (state_u3, score)
        }
      } 
    """
    scores = {}

    # Base Cases: Do not need to account
    # Reason: pi(0 , START) = 1 otherwise 0
    # when we take log, they become 0.

    # START state to state 1
    scores[0] = {}

    for state_v in states:
        # Transition Probability
        trans_frac = get_transition_parameters(transition_dict, "START", state_v)
        if trans_frac != 0:
            trans = math.log(trans_frac)
        else:
            trans = smallest
        
        # if the word does not exist, assign special token
        if sentence[0] not in observations:
            obs = "#UNK#"
        else:
            obs = sentence[0]

        # Emission Probability
        if ((obs in emission_dict[state_v]) or (obs == "#UNK#")): 
            emis_frac = estimate_emission_param(emission_dict, obs, state_v)
            emis = math.log(emis_frac)
        else:
            emis = smallest
        
        start = trans + emis
        scores[0][state_v] = ("START", start)

    scores_copy = copy.deepcopy(scores)
    #print(scores_copy)
    
    # State 1 to n
    for i in range(1, n):
        scores[i] = {}
        scores_copy[i] = {}
        for state_v in states:
            findmax = []
            for state_u in states:
                # Transition Probability
                trans_frac = get_transition_parameters(transition_dict, state_u, state_v)
                if trans_frac != 0:
                    trans = math.log(trans_frac)
                else:
                    trans = smallest
                
                # if the word does not exist, assign special token
                if sentence[i] not in observations:
                    v = "#UNK#"
                else:
                    v = sentence[i]

                # Emission Probability
                if ((v in emission_dict[state_v]) or (v == "#UNK#")): 
                    emis_frac = estimate_emission_param(emission_dict, v, state_v)
                    emis = math.log(emis_frac)
                else:
                    emis = smallest
              
                if i == 1 :
                  currentscore = scores[i-1][state_u][1] + trans + emis
                  findmax.append(currentscore)
                else:
                  currentscores = [[scores[i-1][state_u][m][1] for m in range(N)][j] + trans + emis for j in range(N)] # currentscores = [bestscore, 2nd bestscore, 3rd bestscore]
                  for score in currentscores:
                    findmax.append(score)
            # findmax = [bestscore, 2nd bestscore, 3rd bestscore,bestscore, 2nd bestscore, 3rd bestscore,bestscore, 2nd bestscore, 3rd bestscore,...,bestscore, 2nd bestscore, 3rd bestscore]  
            
            # ARGMAX
            # code to find N highest scores in findmax
            # since there are nT scores, we have to argmax over ALL these scores
            # ans is a list that holds the N highest scores
            # eg. ans = [highest, 2nd highest, 3rd highest, .... , N highest]
            # state_ans is a list that holds the N best states, with decrementing "bests state" from left to right
            # eg. state_ans = [best state, 2nd best state, ..... , N best state]

            ans = [] 
            state_ans = []
            findmax_copy = copy.deepcopy(findmax)
            for m in range(N):
                ans.append(max(findmax_copy))
                state_ans.append(states[findmax.index(ans[m]) // N])
                findmax_copy[findmax.index(ans[m])] = -999999999.999
            
            # store nested tuple of N best states ((best state, score),(2nd best state, score),(3rd best state, score))
            scores[i][state_v] = tuple((state_ans[m], ans[m]) for m in range(N))
            
    #print('scores',scores)
    # STOP STATE
    scores[n] = {}
    scores_copy[n] = {}
    stopmax = []

    for state_u in states:
        # Transition Probability
        trans_frac = get_transition_parameters(transition_dict, state_u, "STOP")
        if trans_frac != 0:
            trans = math.log(trans_frac)
        else:
            trans = smallest

        #stopscore = [[scores[n-1][state_u][m][1] for m in range(N)][j] + trans + emis for j in range(N)]
        

        if(type(scores[n-1][state_u][0])==tuple):
            stopscore = [[scores[n-1][state_u][m][1] for m in range(N)][j] + trans + emis for j in range(N)]
        else:
            t=scores[n-1][state_u]
            stopscore = [t[1]+ trans + emis]    


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
    for i in range(N):
        stop.append(max(stopmax_copy))
        state_ans.append(states[stopmax.index(stop[i]) // N])
        stopmax_copy[stopmax.index(stop[i])] = -999999999.999
    scores[n][state_u] = tuple((state_ans[m], stop[m]) for m in range(N))
    
      
    # Backtracking path
    # N_bestPaths is list of N lists, that holds N best paths in decreasing order.
    # eg. N_bestPaths = [best path, 2nd best path, ... , N best path]
    # lasts is a list to trac k the last state of each of the N best paths. 
    # eg. lasts = [best path's last, 2nd best path's last, ... , N best path's last]
    N_bestPaths = []
    lasts = [] 
    for i in range(N):
      path = ["STOP"]
      last = list(scores[n].values())[0][i][0]
      lasts.append(last)
      path.insert(0, last)
      N_bestPaths.append(path)
    
    for i in range(N):
        for k in range(n-1, -1, -1):
            if k == 0:
                last = scores[k][N_bestPaths[i][0]][0] 
            else:
                last = scores[k][N_bestPaths[i][0]][0][0]
            N_bestPaths[i].insert(0, last)
    
    return N_bestPaths[N-1]


train_obs, emission_counts = emission_counting('./RU/train')
transition_counts = transition_counting('./RU/train')
states = list(transition_counts.keys())
sequence = data_dump('./RU/dev.in')
sequences = []

for i in sequence:
    for j in i: 
        sequences.append(j)

viterbi_outputs = viterbi_5(5, emission_counts, transition_counts,  sequence, states)

# with open('./ES/dev.in', "r", encoding="utf8") as f:
#             lines = f.readlines()

# with open('./ES/dev.p3.out', "w", encoding="utf8") as g:
#     k = 0
#     num_lines = 0
#     for j in range(len(lines)):
#         word = lines[j].strip()

        
#         if (word != ""):
#             path = viterbi_outputs[k][1]
#             g.write(word + " " + path)
#             g.write("\n")
#             k += 1   
            
#         else:
#             g.write("\n")

if __name__ == '__main__':
    part4_dataset = ["RU"]

    for i in part4_dataset:
        """
        train = "Data/{folder}/train".format(folder = i)
        evaluation = "Data/{folder}/dev.in".format(folder = i)
        """

        root_dir = './'
        
        train = root_dir + "{folder}/train".format(folder = i)
        evaluation = root_dir + "{folder}/dev.in".format(folder = i)
        
        # training
        transition_tracker = transition_counting(train)

        obs_all, emission_tracker = emission_counting(train)
        
        # evaluation
        with open(evaluation, "r", encoding= 'utf8') as f:
            # readlines() returns a list containing each line in the file as a list item
            # each line is a word
            lines = f.readlines()
        
        # track each sentence's prediction labels
        # each word's prediction label will be an element of this list
        sentence = []
        
        # list containing all prediction labels
        # sentences are separated with element "\n" in between
        all_prediction = []
        #print(i)
        # initialise N
        N = 5
        # each line is a word
        for line in lines:        
            if line != "\n":
                line = line.strip()
                sentence.append(line)
            else:
                sentence_prediction = viterbi_5(N, emission_tracker, transition_tracker, obs_all, sentence)
                sentence_prediction.remove("START")
                sentence_prediction.remove("STOP")
                all_prediction = all_prediction + sentence_prediction
                all_prediction = all_prediction + ["\n"]
                sentence = []
        
        assert len(lines) == len(all_prediction)
        # create output file
        with open('./' + "{folder}/dev.p3.out".format(folder = i), "w", encoding='utf8') as g:
            for j in range(len(lines)):
                word = lines[j].strip()
                if word != "\n":
                    tag = all_prediction[j]
                    if(tag != "\n"):
                        g.write(word + " " + tag)
                        g.write("\n")
                    else:
                        g.write("\n")

    print("done")