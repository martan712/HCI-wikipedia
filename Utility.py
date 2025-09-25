import pandas as pd


def compute_click_count_from_transitions(transitions):
    return  transitions.groupby(['Current_A', 'Next_A_prime', 'Goal_G']).size().reset_index(name='N_click')


def compute_encounters_from_transitions(transitions):
    return transitions.groupby(['Current_A', 'Goal_G']).size().reset_index(name='N_encounter')

def compute_out_degree(edges):
    return edges.groupby('Current_A').size().reset_index(name='Out_Degree')
    
def compute_posteriors(transitions, edges):
    ALPHA = 0.1
    
    # Compute statistics over dataset
    click_counts = compute_click_count_from_transitions(transitions)
    encounter_counts = compute_encounters_from_transitions(transitions)
    La_counts = compute_out_degree(edges)
    
    # 1. Merge the click and encounter counts
    posteriors_df = pd.merge(click_counts, encounter_counts, on=['Current_A', 'Goal_G'], how='left')

    # 2. Merge with the out-degree (La) approximation
    posteriors_df = pd.merge(posteriors_df, La_counts, on='Current_A', how='left')

    # 3. Calculate the Posterior Click Probability (P*)
    posteriors_df['P_star'] = (
        posteriors_df['N_click'] + ALPHA
    ) / (
        posteriors_df['N_encounter'] + ALPHA * posteriors_df['Out_Degree']
    )
    
    # return dataframe
    return posteriors_df