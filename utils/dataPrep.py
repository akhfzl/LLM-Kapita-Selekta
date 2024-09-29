import pandas as pd, os
import numpy as np

class DataPreparation:
    def __init__(self, pathFolder=None):
        self.folder = pathFolder

    # membuka folder dan list
    def openFolder(self, folderpath=None):
        if not folderpath:
            folderpath = self.folder 

        try:
            return os.listdir(folderpath)
        except:
            print('The pathFolder of class object should be filled')
    
    # memilih file small or large dataset
    def selectSmallOrLarge(self, isSmallOrLarge='large'):
        checkFolder = self.openFolder('/kaggle/input/llm-pretrained/new_dataset')

        if isSmallOrLarge == 'large' and 'large_corpus.csv' in checkFolder:
            return pd.read_csv(os.path.join('/kaggle/input/llm-pretrained/new_dataset', 'large_corpus.csv'))
        
        elif isSmallOrLarge == 'small' and 'small_corpus.csv' in checkFolder:
            return pd.read_csv(os.path.join('/kaggle/input/llm-pretrained/new_dataset', 'small_corpus.csv'))
        
        else:
            print('Please check your file')
            pass
    
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
