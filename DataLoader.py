import pandas as pd
import networkx as nx

class DataLoader():
    
    def get_file_in_dir(self,path):
        if self.data_dir is None:
            return path
        return f"{self.data_dir}/{path}"
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        
    def load_paths(self, filename):
        paths = pd.read_csv(self.get_file_in_dir(filename), sep="\t")
        paths.columns = ["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"]

        # Split the path string into a list of articles
        paths['path'] = paths['path'].str.split(';')

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
        pagerank_df = pagerank_series.reset_index()
        pagerank_df.columns = ['Current_A', 'PageRank']
        self.pagerank = pagerank_df
        
    def load_edges(self, filename):
        edges = pd.read_csv(filename, sep="\t")
        edges.columns = ["Current_A","Next_A"]
        self.edges = edges
        
        self.load_pagerank()