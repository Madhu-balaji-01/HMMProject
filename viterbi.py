import math

def viterbi(emission_counts, transition_counts, observations):
    # Retrieving a list of all possible states
    states = [transition_counts.keys()]
    states.append('STOP') # START is already in the list, no need to append

    # 'observations' are of the form [['sentence1_word1', 'sentence1_word2', 'sentence1_word3'], ['sentence2_word1']]
    for observation in observations: 
        # Initializing pi scores
        pi_scores = []
        pi_scores.append({'START' : 1}) 
        # Sentence length
        n = len(observation)
        path_list = []

        for j in range(n): 
            pi_scores[j+1] = {}
            if j==0:
                for u in states:
                    # emission_param = get_emission_params(emission_counts, u, observation[j+1])
                    emission_param = 0.3
                    # To avoid underflow issue, we take log of emission and transition params
                    log_emission_param = math.log(emission_param)
                    transition_param = get_transition_params(transition_counts, 'START', u)
                    # transition_param = 0.8
                    log_transition_param = math.log(transition_param)
                    # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
                    score_list.append(pi_scores[j]['START'] + log_emission_param + log_transition_param)

            else:
                for u in states:
                    # emission_param = get_emission_params(emission_counts, u, observation[j+1])
                    emission_param = 0.3
                    # To avoid underflow issue, we take log of emission and transition params
                    log_emission_param = math.log(emission_param)

                    score_list = []
                    for v in states:
                        # transition_param = get_transition_params(transition_counts, v, u)
                        transition_param = 0.8
                        log_transition_param = math.log(transition_param)
                        # Since we took log, we add pi_score + emission probablity + transition probability (not multiply)
                        score_list.append(pi_scores[j][v] + log_emission_param + log_transition_param)

                    max_score = max(score_list)
                    pi_scores[j+1][u] = max_score
                path_list.append(states[score_list.index(max_score)])
                print(path_list)
        
        # STOP state



            
        
    

# viterbi(0, {'START': {'O': 1918, 'B-negative': 27, 'B-positive': 110, 'B-neutral': 10}}, [['a', 'b']] )
