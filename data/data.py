"""
    File to load dataset based on user control from main file
"""
from data.SBMs import SBMsDataset
from data.TSP import TSPDataset
from data.WikiCS import WikiCSDataset
from data.CO import CODataset


def LoadData(data_dir, name, split):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    return CODataset(data_dir, name, split)
