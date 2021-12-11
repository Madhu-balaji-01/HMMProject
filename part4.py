# This code is an implementation of Structured Perceptron algorithm.
# Reference - https://taehwanptl.github.io/lectures/lecture_05_04.pdf

import re
import string
from typing import NewType
from utils import *
from part1_final import *
from part2 import *
from collections import Counter, defaultdict

def train_perceptron(train_data_path, lr = 0.1):
    # From data_dump in utils.py we get data in the form of 
    # [['sentence1_word1 label1', 'sentence1_word2 label2'],
    #  ['sentence2_word1 label1', 'sentence2_word2 label2']] 
    train_data = data_dump(train_data_path)

    # Initializing feature weight dictionaries
    feature_weights = defaultdict(float)
    total_weights = defaultdict(float)

    # Retrieve a list of all possible states
    states = set()
    for sentence in train_data:
        for word_tag in sentence:
            word_tag = word_tag.split()
            states.add(word_tag[1])
    i=0
    for sentence in train_data:
        # print(sentence)
        # Split into observations and tags
        observations, tags = [], []
        for i in range(len(sentence)):
            # print(i)
            parts = sentence[i].split() 
            observations.append(parts[0])
            tags.append(parts[1])
        
        # First train Viterbi and get its predictions
        viterbi_predictions = viterbi_perceptron(feature_weights, states, sentence)
        # print(viterbi_predictions)

        true_feature_vec = get_feature_dict(observations, tags)
        predicted_feature_vec = get_feature_dict(observations, viterbi_predictions)

        # PERCEPTRON UPDATE RULE 
        if viterbi_predictions != tags:
            # Update weights based on ground truth features
            feature_weights, total_weights = update_weights(lr,
                                                            true_feature_vec, 
                                                            feature_weights, 
                                                            total_weights)
            updated1_dict = feature_weights
            # print('updated features', feature_weights)
            # Penalize wrong predictions
            for feat, count in predicted_feature_vec.items():
                feature_weights[feat] = feature_weights[feat] - lr * count
            updated_dict = feature_weights
        # i+=1
        # if i==1:
        #     print(updated1_dict==updated_dict)
        #     break
        
    return feature_weights, states


def get_feature_dict(x, y):
    feature_dict = Counter()
    for i, (word, tag) in enumerate(zip(x, y)):
        # If this is the first word, previous state is START
        if i == 0:
            prev_tag = "START" 
            prev_word = "None"
        else: 
            prev_tag = y[i-1]
            prev_word = x[i-1]
        
        if i == (len(y) - 1):
            next_word = "END"
        else:
            next_word = x[i+1]
        feature_dict.update(generate_features(word, tag, prev_word, next_word, prev_tag))
    return feature_dict


def generate_features(observation, tag, prev_word, next_word,prev_tag):
    # Here we define a few features for the data (refer to lecture in link above)
    lower_case = observation.lower()
    suffix_1 = lower_case[-1]
    prefix_1 = lower_case[0]
    suffix_2 = lower_case[-2:]
    prefix_2 = lower_case[:2]
    suffix_3 = lower_case[-3:]
    prefix_3 = lower_case[:3]
   
    features = [f"TAG_{tag}",
                f"TAG_2ORDER_{prev_tag}_{tag}",
                f"WORD_{observation}",
                # f"WORD_BIGRAM_{prev_word}_{observation}",
                # f"WORD_TRIGRAM_{prev_word}_{observation}_{next_word}"
                f"PUNCTUATION_{observation in string.punctuation}",
                f"LOWERCASE+TAG_{lower_case}_{tag}",
                # f"SUFFIX_1_{suffix_1}_{tag}",
                # f"PREFIX_1_{prefix_1}_{tag}",
                # f"SUFFIX_2_{suffix_2}_{tag}",
                # f"PREFIX_2_{prefix_2}_{tag}",
                f"SUFFIX_3_{suffix_3}",
                f"PREFIX_3_{prefix_3}",
                f"SUFFIX_TAG_{suffix_3}_{tag}",
                f"SUFFIX_TAG_PREVTAG_{suffix_3}_{tag}_{prev_tag}",
                f"PREFIX_TAG_{prefix_3}_{tag}",
                f"PREFIX_TAG_PREVTAG_{prefix_3}_{tag}_{prev_tag}"]
    # features = [f"PREFIX2+2TAGS_{prefix_2}_{prev_tag}_{tag}",
    #         f"PREFIX3+2TAGS_{prefix_3}_{prev_tag}_{tag}",
    #         f"DASH_{'-' in observation}_{tag}",
    #         f"WORD_LOWER+TAG_{lower_case}_{tag}",
    #         f"UPPER_{observation[0].isupper()}_{tag}",
    #         f"PREFIX2+TAG_{prefix_2}_{tag}",
    #         f"SUFFIX3+2TAGS_{suffix_3}_{prev_tag}_{tag}",
    #         f"WORD_LOWER+TAG_BIGRAM_{lower_case}_{tag}_{prev_tag}",
    #         f"SUFFIX3+TAG_{suffix_3}_{tag}",
    #         f"SUFFIX3_{suffix_3}",
    #         f"SUFFIX2+TAG_{suffix_2}_{tag}",
    #         f"SUFFIX2+2TAGS_{suffix_2}_{prev_tag}_{tag}",
    #         # f"PREV_2_TAGS_{two_previous_tags}"
    #         f"WORD_LOWER+TAG_{lower_case}_{tag}",
    #         f"PREFIX3+TAG_{prefix_3}_{tag}",
    #         f"PREFIX2_{prefix_2}",
    #         f"TAG_{tag}",
    #         f"TAG_BIGRAM_{prev_tag}_{tag}",
    #         f"SUFFIX2_{suffix_2}",
    #         # f"WORDSHAPE_{self._shape(word)}_TAG_{tag}",
    #         f"PREFIX3_{prefix_3}",
    #         f"ISPUNC_{observation in string.punctuation}"
    #         ]
    return features


def update_weights(lr, features, feature_weights, total_weights):
    for feat, count in features.items():
        w = feature_weights[feat]
        if not total_weights[feat]:
            total_weight = 0
        else:
            total_weight = total_weights[feat]

        # Adding current feature weight to total weight and updating it
        total_weight += w
        total_weight += lr * count

        # Updating weights dictionaries
        feature_weights[feat] += lr * count
        total_weights[feat] = total_weight
    
    return feature_weights, total_weights


# Modified Viterbi algorithm using weights to make predictions
# along with Structured Perceptron
def viterbi_perceptron(feature_weights, states, observation):
    # pi_scores = {}
    # pi_scores[0] = {'START' : 1}
    # n = len(observation)
    # ultimate_path = []

    # # print('n', n)
    # # print(observation)
    # for j in range(n):
    #     pi_scores[j+1] = {}
    #     # Transition from 'START' to first state and emission of first state to first word 
    #     if j==0:
    #         for u in states:
    #             if n>1:
    #                 next_word = observation[1]
    #             else:
    #                 next_word = "None"
    #             features = generate_features(observation[0], u, 
    #                                         prev_word="None",
    #                                         next_word = next_word,
    #                                         prev_tag="START")
    #             current_weight = sum((feature_weights[feat] for feat in features))
    #             pi_scores[j+1][u] = current_weight
        
    #     else:
    #         for u in states:
    #             current_scores = {}    
    #             score_list = [] 
    #             for v in pi_scores[j].keys():
    #                 if j < n-1:
    #                     next_word = observation[j+1]
    #                 else:
    #                     next_word = 'None'
    #                 features = generate_features(observation[j], u, 
    #                                         prev_word=observation[j-1],
    #                                         next_word = next_word,
    #                                         prev_tag=v)
    #                 current_weight = sum((feature_weights[feat] for feat in features))

    #                 # current_scores[v] = pi_scores[j][v] + feature_weights
    #                 score_list.append(pi_scores[j][v] + current_weight)

    #             # Storing the maximum score over all v's
    #             max_score = max(score_list)
    #             pi_scores[j+1][u] = max_score
            
    # # Transition from last state to 'STOP'
    # pi_scores[n+1] = {}
    # for u in pi_scores[n].keys():
    #     current_weight = feature_weights[(u, "STOP")]
    #     pi_scores[n+1][u] = pi_scores[n][u] +  current_weight
    
    # # Backward algorithm
    # back_tracker = []
    
    # for u_star in range(n,0,-1):
    #     temp = max(pi_scores[u_star], key=pi_scores[u_star].get)
    #     back_tracker.insert(0,temp)
        
    # ultimate_path.append(back_tracker) 
    # # print(ultimate_path)
    # return ultimate_path

    n = len(observation)
    pi_scores = { 1: {} }
    state_tracker = { 1: {} }
    for state in states:
        if n>1:
            next_word = observation[1]
        else:
            next_word = "None"
        features = generate_features(observation[0], state, 
                                    prev_word="None",
                                    next_word = next_word,
                                    prev_tag="START")
        current_weights = sum((feature_weights[x] for x in features))
        pi_scores[1][state] = current_weights
        state_tracker[1][state] = "START"
    
    # Move forward recursively
    for j in range(1,n):
        # print('j', j)
        pi_scores[j+1] = {}
        state_tracker[j+1] = {}
        for u in states:
            score_dict = {}
            for v in pi_scores[j].keys():
                if j < n-1:
                    next_word = observation[j+1]
                else:
                    next_word = 'None'

                features = generate_features(observation[j], u, 
                                            prev_word=observation[j-1],
                                            next_word = next_word,
                                            prev_tag=v)
                current_weights = sum((feature_weights[x] for x in features))
                # print(pi_scores)
                score_dict[v] = pi_scores[j][v] + current_weights
                
            max_state = max(score_dict, key=score_dict.get)
            pi_scores[j+1][u] = score_dict[max_state]
            state_tracker[j+1][u] = max_state
        
    
    # Transition to STOP
    score_dict = {}
    for u in pi_scores[n].keys():
        current_weights = feature_weights[(u, "STOP")]
        score_dict[u] = pi_scores[n][u] + current_weights
    
    # Best y_n
    last_y = max(score_dict, key=score_dict.get)
    predicted = [last_y]

    # Backtrack
    for k in range(n , 0, -1):
        next_state = predicted[-1]
        predicted.append(state_tracker[k][next_state])
    predicted.reverse()
    
    return predicted[1:]


def get_predictions(test_data_path, test_output_path, train_path):
    # Train model first
    trained_feature_weights, states = train_perceptron(train_path)
    observations = data_dump(test_data_path)

    viterbi_outputs = []
    for observation in observations:
        output = viterbi_perceptron(trained_feature_weights, states, observation)
        viterbi_outputs.append(output)
    # print('out',viterbi_outputs)
    with open(test_data_path, "r", encoding="utf8") as f:
        lines = f.readlines()
                
    with open(test_output_path, "w", encoding="utf8") as g:
        k = 0
        num_lines = 0
        for j in range(len(lines)):
            word = lines[j].strip()   
            
            if (word != ""):
                path = viterbi_outputs[k][j-num_lines]
                # print(path)
                g.write(word + " " + path)
                g.write("\n")
            else:
                k+=1
                num_lines = j + 1
                g.write("\n")
    print('DONE!')


if __name__=="__main__":
    get_predictions(test_data_path='./ES/dev.in',
                    test_output_path='./ES/dev.p4.out',
                    train_path='./ES/train')