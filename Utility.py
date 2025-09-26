import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_click_count_from_transitions(transitions):
    return  transitions.groupby(['Current_A', 'Next_A', 'Goal_G']).size().reset_index(name='N_click')


def compute_encounters_from_transitions(transitions):
    return transitions.groupby(['Current_A', 'Goal_G']).size().reset_index(name='N_encounter')

def compute_out_degree(edges):
    outd= edges.groupby('Current_A').size().reset_index(name='Out_Degree')
    return outd

def compute_posteriors(transitions, edges, ALPHA=0.1):
    # Compute statistics over dataset
    click_counts = compute_click_count_from_transitions(transitions)
    encounter_counts = compute_encounters_from_transitions(transitions)
    La_counts = compute_out_degree(edges)
    
    # 1. Merge the click and encounter counts
    posteriors_df = pd.merge(click_counts, encounter_counts, on=['Current_A', 'Goal_G'], how='left')


    posteriors_df = pd.merge(posteriors_df, La_counts, on='Current_A', how='left')

    # 3. Calculate the Posterior Click Probability (P*)
    posteriors_df['P_star'] = (
        posteriors_df['N_click'] + ALPHA
    ) / (
        posteriors_df['N_encounter'] + ALPHA * posteriors_df['Out_Degree']
    )
    
    # return dataframe
    return posteriors_df

def compute_path_distance(
        path,
        goal,
        posteriors,
        pageranks
    ):
    logs = []

    for j in range(len(path) - 1):
        a_current = path[j]
        a_next = path[j+1]
        
        # Get the probability from the model
        P_star = posteriors.loc[(a_current,a_next, goal)].values[0]

        if P_star <= 0:
            # Handle impossible/invalid transitions
            return None 

        # Compute -log(P*) for this single transition
        neg_log_p = np.log(P_star)
        logs.append(neg_log_p)

    # Sum the individual negative log-likelihoods
    sum_neg_logs = -sum(logs)

    # --- 2. Compute Goal Prior Cost (Negative Log-PageRank) ---
    # Term 2: -log PageRank(g)
    if goal in pageranks.index:
        PR_g = pageranks.loc[goal]
    else:
        PR_g = np.inf
        

    pagerank_cost = -np.log(PR_g)
    
    # --- 3. Compute Total Cost ---
    
    total_cost = sum_neg_logs / pagerank_cost
    
    return {"start":path[0] , "goal":goal, "distance":total_cost}

def compute_sub_path_distances(
        path,
        goal,
        posteriors,
        pageranks
    ):

    path_distances = []
    for i in range(len(path)-1):
        path_dist = compute_path_distance(path[i:], goal, posteriors, pageranks)
        if type(path_dist) is dict:
            path_distances.append(path_dist)
        
    return path_distances

def compute_path_specific_distances(paths, posteriors, pageranks):
    transition_distances = []
    for _, path in tqdm(paths.iterrows(), total=len(paths), desc="Processing Paths"):
        sub_path_dists = compute_sub_path_distances(path["path"], path["Goal"], posteriors, pageranks)

        transition_distances+=sub_path_dists
        

    full_distances_df = pd.DataFrame(transition_distances)
    return full_distances_df