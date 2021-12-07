from utils import *

def transition_counting(path):
    
    temp_list = data_dump(path)
    
    start_state = "START"
    stop_state = "STOP"
    
    state_i1 = start_state
    end_sentence = False
    
    transition_dict = {}
    
    for i in temp_list:
        
        if end_sentence == False:
            
            for j in i:
            
                #if j!= (i[len(i) - 1]):
                
                word = j
                word = word.rsplit(" ")
                state_i2 = word[1]
            
                if state_i1 not in transition_dict:
                    state_i1_dict = {}
                else:
                    state_i1_dict = transition_dict[state_i1]
            
                if state_i2 in state_i1_dict:
                    state_i1_dict[state_i2] += 1
                else:
                    state_i1_dict[state_i2] = 1
                    
                transition_dict[state_i1] = state_i1_dict
                state_i1 = state_i2
                
                if j == i[len(i) - 1]:
                    end_sentence = True
                    
        if end_sentence == True:
          
            if state_i1 not in transition_dict:
                state_i1_dict = {}
            else:
                state_i1_dict = transition_dict[state_i1]

            state_i2 = stop_state
            
            if state_i2 in state_i1_dict:
                state_i1_dict[state_i2] += 1
            else:
                state_i1_dict[state_i2] = 1
                
            transition_dict[state_i1] = state_i1_dict
            state_i1 = start_state
            end_sentence = False
                
    return transition_dict

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