import logging
import networkx as nx

log = logging.getLogger(__name__)

class DiGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

