import pandas as pd, os
import numpy as np

class DataPreparation:
    def __init__(self, pathFolder):
        self.folder = pathFolder

    # membuka folder dan list
    def openFolder(self):
        try:
            return [f'{self.folder}/{fname}' for fname in os.listdir(self.folder)]
        except:
            print('The pathFolder of class object should be filled')
    
    # membaca file
    def fileAsDataframe(self, filename, filetype='json'):
        df = pd.read_json(filename)

        if filetype != 'json':
            df = pd.read_csv(filename)

        return df 
    
    # membaca semua file json dari corpus
    def readAllData(self, listdirs):
        df = pd.DataFrame()

        for fname in listdirs:
            df_other = pd.read_json(fname)

            df = pd.concat([df, df_other], ignore_index=True)

        return df
    
    # memilih fitur yang dibutuhkan
    def feature_selection_byKey(self, list_key, df):
        return df[list_key]
