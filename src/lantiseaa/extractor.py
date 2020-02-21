from abc import ABC, abstractmethod


class TSFeatureExtractor(ABC):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def extract_features(self, ts):
        pass

    @abstractmethod
    def select_relevant_features(self, X, y, fdr_level=0.001):
        pass
