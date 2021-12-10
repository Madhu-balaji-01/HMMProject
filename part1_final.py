import numpy as np
from utils import *

def emission_counting(path):
    # get nested list from input file
    seq = data_dump(path)
    
    # track emission count
    # key: state y, value: nested dictionary with key = obs x and value = frequency of this specific obs x
    emission_dict = {}
    end_sentence = False
    observations = set()
    # state_i1 is observed y
    # state_i2 is state x

    for i in seq:

        if end_sentence == False:
            
            for j in i:
            
                #if j!= (i[len(i) - 1]):
                
                word = j
                word = word.rsplit(" ")
                
                if(len(word) != 2):
                    state_i1 = word[2]
                    state_i2 = word[0] + ' ' + word[1]
                    
                else:
                    state_i1 = word[1]
                    state_i2 = word[0]
                observations.add(state_i2)
            
                if state_i1 not in emission_dict:
                    state_i1_dict = {}
                else:
                    state_i1_dict = emission_dict[state_i1]
            
                if state_i2 in state_i1_dict:
                    state_i1_dict[state_i2] += 1
                else:
                    state_i1_dict[state_i2] = 1
                    
                emission_dict[state_i1] = state_i1_dict
                #state_i1 = state_i2
                
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
                
    return observations, emission_dict

    

def estimate_emission(emission_dict, x, y):

    state_dict = emission_dict[y]
    

    if x in state_dict:
        numerator = state_dict[x]
    else:
        numerator = 0
    
    denominator = sum(state_dict.values())
    
    return numerator / denominator

def estimate_emission_param(emission_dict, x, y, k = 1):
    state_dict = emission_dict[y]
    
    denominator = sum(state_dict.values()) + k
    

    if (x != "#UNK#") and (x in state_dict):
        numerator = state_dict[x]
    elif x=='#UNK#': 
        numerator = k
    else:
        numerator = 0

    return numerator / denominator

def get_transition_parameters(transition_dict, state_i1, state_i2):
    
    if state_i1 not in transition_dict:
        fraction = 0   
    else:
        state_i1_dict = transition_dict[state_i1]
    
    if state_i2 in state_i1_dict:
        numerator = state_i1_dict[state_i2]
    else:
        numerator = 0
        
    denominator = sum(state_i1_dict.values())
    fraction = numerator / denominator
    
    return fraction

def labelling(inp,emission_dict,observations):
    
    return_list = []
    
    for i in inp:
        
        #temp_list = []
        
        for j in range(len(i)):
            prob = 0
            state = ""
        
            for y in emission_dict:
            
                if i[j] not in observations:
                    i[j] = "#UNK#"
                
                if ((i[j] == "#UNK#") or (i[j] in emission_dict[y])):
                
                    if estimate_emission_param(emission_dict,i[j],y,1) > prob:
                        prob = estimate_emission_param(emission_dict,i[j],y,1)
                        state = y
                        
            #temp_list.append(state)
            return_list.append(state)
            
            if j == (len(i) - 1):
                return_list.append('\n')
            
        #return_list.append(temp_list)
                        
    return return_list


def final_answers_part1(data_file):
    
    train = "{folder}/train".format(folder = data_file)
    test =  "{folder}/dev.in".format(folder = data_file)
    
    observations, emission_dict = emission_counting(train)
    
    test_sentences = data_dump(test)
    
    labels = labelling(test_sentences,emission_dict,observations)
    
    return(labels)
    
    