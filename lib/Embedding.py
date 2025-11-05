from abc import ABC, abstractmethod

class Embedding(ABC):
    def __init__(self, rep):
        self.rep = rep
    
    @abstractmethod
    def score(query):
        pass

    
