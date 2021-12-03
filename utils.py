import random as rand
import os

def data_dump(path):
    with open(path, "r", encoding="cp437", errors='ignore') as f:
        
        sequence_list = []
        
        for entry in f.read().split("\n\n")[:-1]:
            sequence_list.append(entry.split("\n"))  
            
        return sequence_list

def data_dump_split(path):
    with open(path, "r", encoding="cp437", errors='ignore') as f:
        
        sequence_list = []
        temp1 = []
        temp2 = []
        
        for entry in f.read().split("\n\n")[:-1]:
            temp1.append(entry.split("\n"))
            
        for sequence in temp1:
            for i in sequence:
                temp2.append([i.split()])
            
        for sequence in temp2:
            for i in sequence:
                sequence_list.append([i[0], i[1]])
                
        return sequence_list
    
def data_dump_shuffle(path):
    with open(path, "r", encoding="cp437", errors='ignore') as f:
        
        sequence_list = []
        
        for entry in f.read().split("\n\n")[:-1]:
            sequence_list.append(entry.split("\n"))
        
        rand.shuffle(sequence_list)
            
        return sequence_list