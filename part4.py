# This code is an implementation of Structured Perceptron algorithm.
# Reference - https://taehwanptl.github.io/lectures/lecture_05_04.pdf

from os import stat
import string
from utils import *
from collections import Counter, defaultdict

# Initializing global variables
feature_weights = defaultdict(float)  # Feature weight dictionary
states = []


def train_perceptron(train_data_path, lr = 0.2):
    global states, feature_weights
    # From data_dump in utils.py we get data in the form of 
    # [['sentence1_word1 label1', 'sentence1_word2 label2'],
    #  ['sentence2_word1 label1', 'sentence2_word2 label2']] 
    train_data = data_dump(train_data_path)

    # Retrieve a list of all possible states
    states_set = set()
    for sentence in train_data:
        for word_tag in sentence:
            parts = word_tag.split()
            # Handling the '. ..' cases in RU dataset
            if len(parts) == 2:
                states_set.add(parts[1])
            else:
                states_set.add(parts[2])
    states = list(states_set)

    for sentence in train_data:
        # print(sentence)
        # Split into observations and tags
        observations, tags = [], []
        for i in range(len(sentence)):
            parts = sentence[i].split() 
            # Handling the '. ..' cases in RU dataset
            if len(parts) == 2:
                observations.append(parts[0])
                tags.append(parts[1])
            else:
                observations.append(parts[0] + parts[1])
                tags.append(parts[2])
            
        # First train Viterbi and get its predictions
        viterbi_predictions = viterbi_perceptron(observations)
         
        true_feature_vec = get_feature_dict(observations, tags)
        predicted_feature_vec = get_feature_dict(observations, viterbi_predictions)

        # PERCEPTRON UPDATE RULE 
        if viterbi_predictions != tags:
            # Update weights based on ground truth features
            update_weights(lr,true_feature_vec)
            # print('updated features', feature_weights)

            # Penalize wrong predictions
            for feat, count in predicted_feature_vec.items():
                feature_weights[feat] = feature_weights[feat] - lr * count
            
      
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
        # If this is the last word, next word is 'END'
        if i == (len(x) - 1):
            next_word = "END"
        else:
            next_word = x[i+1]
        
        feature_dict.update(generate_features(word, tag, prev_word, next_word, prev_tag))
    return feature_dict


def generate_features(observation, tag, prev_word, next_word,prev_tag):
    # Here we define a few features for the data (refer to lecture in link above)
    lower_case = observation.lower()
    suffix_3 = lower_case[-3:]
    prefix_3 = lower_case[:3]
   
    features = [
            f"PREFIX3+2TAGS_{prefix_3}_{prev_tag}_{tag}",
            f"DASH_{'-' in observation}_{tag}",
            f"WORD_LOWER+TAG_{lower_case}_{tag}",
            f"UPPER_{observation[0].isupper()}_{tag}",
            # f"PREFIX2+TAG_{prefix_2}_{tag}",
            f"SUFFIX3+2TAGS_{suffix_3}_{prev_tag}_{tag}",
            f"WORD_LOWER+TAG_BIGRAM_{lower_case}_{tag}_{prev_tag}",
            f"SUFFIX3+TAG_{suffix_3}_{tag}",
            f"SUFFIX3_{suffix_3}",
            f"WORD_LOWER+TAG_{lower_case}_{tag}",
            f"PREFIX3+TAG_{prefix_3}_{tag}",
            f"TAG_{tag}",
            f"TAG_BIGRAM_{prev_tag}_{tag}",
            f"PREFIX3_{prefix_3}",
            f"ISPUNC_{observation in string.punctuation}"
            ]
    return features


def update_weights(lr, features):
    global feature_weights
    for feat, count in features.items():
        # Updating weights dictionaries
        feature_weights[feat] += lr * count


# Modified Viterbi algorithm using weights to make predictions
# along with Structured Perceptron
def viterbi_perceptron(observation):
    global feature_weights, states
    pi_scores = {}
    state_tracker = {}
    n = len(observation)
    ultimate_path = []

    for j in range(n):
        pi_scores[j+1] = {}
        state_tracker[j+1] = {}
        # Transition from 'START' to first state and emission of first state to first word 
        if j==0:
            for u in states:
                if n>1:
                    next_word = observation[1]
                else:
                    next_word = "None"
                features = generate_features(observation[0], u, 
                                            prev_word="None",
                                            next_word = next_word,
                                            prev_tag="START")
                current_weight = sum((feature_weights[feat] for feat in features))
                pi_scores[j+1][u] = current_weight
                state_tracker[j+1][u] = "START"
        
        else:
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
                    current_weight = sum((feature_weights[feat] for feat in features))
                    score_dict[v] = pi_scores[j][v] + current_weight

                # Storing the maximum score and state over all v's 
                max_state = max(score_dict, key=score_dict.get)
                pi_scores[j+1][u] = score_dict[max_state]
                state_tracker[j+1][u] = max_state
    
    # Transition from last state to 'STOP'
    score_dict = {}
    for u in pi_scores[n].keys():
        current_weight = feature_weights[(u, "STOP")]
        score_dict[u] = pi_scores[n][u] +  current_weight
    
    # Backward algorithm
    # Add the state that has max score transitioning to STOP
    y_nstar = max(score_dict, key=score_dict.get)
    ultimate_path = [y_nstar]
    for u_star in range(n,0,-1):
        # Check last added state in path
        next_state = ultimate_path[-1]
        # Add the max state that led to this path
        ultimate_path.insert(0,state_tracker[u_star][next_state])
        
    return ultimate_path[1:]


def get_predictions(test_data_path, test_output_path, train_path, learning_rate):
    # Train model first
    train_perceptron(train_path)
    
    # Predict on dev data
    observations = data_dump(test_data_path)
    viterbi_outputs = []
    for observation in observations:
        output = viterbi_perceptron(observation)
        viterbi_outputs.append(output)
    
    # Write predictions
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
    dataset = input("Please enter dataset ('ES' or 'RU'): ")
    mode = input("Please enter mode (dev or test): ")
    if dataset == "ES":
        test_data_path= f'./ES/{mode}.in'
        test_output_path=f'./ES/{mode}.p4.out'
        train_path='./ES/train'
        lr = 0.1
    elif dataset == "RU":
        test_data_path=f'./RU/{mode}.in'
        test_output_path=f'./RU/{mode}.p4.out'
        train_path='./RU/train'
        lr = 0.25
    else:
        print("Please enter a valid dataset")

    get_predictions(test_data_path,
                    test_output_path,
                    train_path,
                    learning_rate= lr)