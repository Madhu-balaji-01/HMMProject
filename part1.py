import numpy as np
import utils

def emission_counting(path):
    # get nested list from input file
    seq = utils.data_dump(path)
    
    # track emission count
    # key: state y, value: nested dictionary with key = obs x and value = frequency of this specific obs x
    emission_dict = {}
    end_sentence = False
    # state_i1 is observed x
    # state_i2 is state y

    for i in seq:

        if end_sentence == False:
            
            for j in i:
            
                #if j!= (i[len(i) - 1]):
                
                word = j
                word = word.rsplit(" ")
                state_i1 = word[0]
                state_i2 = word[1]
            
                if state_i1 not in emission_dict:
                    state_i1_dict = {}
                else:
                    state_i1_dict = emission_dict[state_i1]
            
                if state_i2 in state_i1_dict:
                    state_i1_dict[state_i2] += 1
                else:
                    state_i1_dict[state_i2] = 1
                    
                emission_dict[state_i1] = state_i1_dict
                state_i1 = state_i2
                
                if j == i[len(i) - 1]:
                    end_sentence = True
                    
        if end_sentence == True:
          
            if state_i1 not in emission_dict:
                state_i1_dict = {}
            else:
                state_i1_dict = emission_dict[state_i1]
            
            if state_i2 in state_i1_dict:
                state_i1_dict[state_i2] += 1
            else:
                state_i1_dict[state_i2] = 1
                
            emission_dict[state_i1] = state_i1_dict
            end_sentence = False
                
    return emission_dict

    

def estimate_emission(emission_dict, x, y):
    # obtain the specific nested dict of state y
    state_dict = emission_dict[y]
    
    # get the value of specific state y -> obs x
    numerator = state_dict.get(x, 0)
    
    # get total counts of state y
    denominator = sum(state_dict.values())
    
    return numerator / denominator

def estimate_emission_param(emission_dict, x, y, k = 1):
    # obtain the specific nested dict of state y
    state_dict = emission_dict[y]
    
    denominator = sum(state_dict.values()) + k
    
    # word token x appears in training set
    if x != "#UNK#":
        numerator = state_dict[x]
    # word token x is special token
    else: 
        numerator = k
    return numerator / denominator