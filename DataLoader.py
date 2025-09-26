import pandas as pd
import networkx as nx

class DataLoader():
    
    def get_file_in_dir(self,path):
        if self.data_dir is None:
            return path
        return f"{self.data_dir}/{path}"
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        
        
    def parse_trace(self,trace_string):
        """
        Parses a trace string where '<' means 'back up one step in the 
        logical (non-'<') path'.

        This implementation uses two lists:
        1. logical_path: The sequence of valid, non-'<' states encountered so far.
        2. final_trace: The resulting sequence with '<' replaced.

        Args:
            trace_string (str): The semicolon-separated trace string.

        Returns:
            str: The processed trace string.
        """
        elements = trace_string.split(';')
        
        # Stores the chronological path of non-'<' states
        logical_path = []
        
        # Stores the final, resulting trace elements
        final_trace = []

        for element in elements:
            if element.strip() == '<':
                # Handle the backstep:
                if len(logical_path) >= 2:
                    # The element to backstep *to* is the second-to-last element 
                    # in the logical path, because the last element is the step 
                    # we are backing *away* from.
                    backstep_target = logical_path[-2]
                    
                    # Append the target to the final trace
                    final_trace.append(backstep_target)
                    
                    # Crucially, remove the last element from the logical path 
                    # as the backstep effectively cancels it out for future backsteps.
                    logical_path.pop() 
                else:
                    # Not enough history to backstep.
                    continue
            else:
                # Not a backstep, append to both the logical path and the final trace
                logical_path.append(element)
                final_trace.append(element)

        return final_trace


    def load_paths(self, filename):
        paths = pd.read_csv(self.get_file_in_dir(filename), sep="\t")
        paths.columns = ["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"]

        # Split the path string into a list of articles
        paths['path'] = paths['path'].map(self.parse_trace)

        # Determine the Goal (G) for each path (the last article)
        paths['Goal'] = paths['path'].apply(lambda x: x[-1])    
        
        self.paths=paths  
        
    def load_pagerank(self):
        G = nx.from_pandas_edgelist(
            self.edges, 
            source='Current_A', 
            target='Next_A', 
            create_using=nx.DiGraph # Directed Graph
        )

        pagerank_scores = nx.pagerank(G, alpha=0.85)
        pagerank_series = pd.Series(pagerank_scores, name='PageRank')
        pagerank_df = pagerank_series
        self.pagerank = pagerank_df
        
    def load_transitions(self):
        transitions = []
        for index, row in self.paths.iterrows():
            path = row['path']
            goal = row['Goal']
            
            # Iterate through the path, from the first article up to the second-to-last
            for i in range(len(path) - 1):
                A = path[i]        # Current Article (a)
                A_prime = path[i+1] # Next Article (a')
                
                transitions.append({
                    'Current_A': A,
                    'Next_A': A_prime,
                    'Goal_G': goal
                }
                )
        transitions_df = pd.DataFrame(transitions)
        self.transitions = transitions_df

    def load_edges(self, filename):
        edges = pd.read_csv( self.get_file_in_dir(filename), sep="\t")
        edges.columns = ["Current_A","Next_A"]
        self.edges = edges
        
        self.load_pagerank()
        self.load_transitions()