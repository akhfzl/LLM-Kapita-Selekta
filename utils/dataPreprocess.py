from utils.dataPrep import DataPreparation
from transformers import GPT2Tokenizer
import os 
from dataset import Datasets

class DataPreprocess(DataPreparation):
    def __init__(self, oldfolder, newfolder):
        # inheritance
        super().__init__(oldfolder) 

        # variable
        self.new_folder = newfolder 
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    # formula memecah splitting
    def wayOfSplitting(self, df):
        test_size =  round(len(df) * 0.2)
        val, train = df[(len(df) - test_size): ], df[:(len(df) - test_size)]
        return train, val

    # memisahkan train-test
    def trainTestSplit(self, isLarge=True):
        typeOfProcess = 'large' if isLarge else 'small'
        self.model_type = typeOfProcess
        
        df = self.selectSmallOrLarge(typeOfProcess)
        selectFeature = df[['isi']].values

        train, val = self.wayOfSplitting(selectFeature)

        return train, val 
    
    # fungsi tokenisasi
    def tokenisasi(self, mydict):
        inputs = self.tokenizer(mydict["text"], padding="max_length", truncation=True, return_tensors='pt')
        inputs['labels'] = inputs['input_ids']  # Set labels to be the same as input_ids
        return inputs

    # main fungsi tokenisasi
    def mainTokenise(self, multiInput):
        datasets = None
        if all(isinstance(item, list) and len(item) > 0 for item in multiInput):
            datasets = Dataset.from_dict({'text': multiInput})
        else:
            flatenning = [item[0] for item in multiInput]
            datasets = Dataset.from_dict({"text": flatenning})
        
        tokenisasi = datasets.map(self.tokenisasi, batched=True)
        tokenisasi.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return tokenisasi
        
    
    def decode_tokenizer(self, ids):
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)