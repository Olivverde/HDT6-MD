import pandas as pd

"""
Universidad del Valle de Guatemala
Author: Oliver Milian
Purpose: CSV loader
"""
class Reader(object):
    # Loads the CSV doc
    def __init__(self, csvDoc):
        self.data = pd.read_csv(csvDoc, encoding= 'unicode_escape')