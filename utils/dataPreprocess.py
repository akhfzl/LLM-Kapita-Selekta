from utils.dataPrep import DataPreparation
from transformers import GPT2Tokenizer
import os 

class DataPreprocess(DataPreparation):
    def __init__(self, oldfolder, newfolder):
        # inheritance
        super().__init__(oldfolder) 

        # variable
        self.new_folder = newfolder 
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # formula memecah splitting
    def wayOfSplitting(df):
        test_size =  round(len(df) * 0.2)
        train, test = df[(len(df) - test_size): ], df[:(len(df) - test_size)]
        return train, test

    # memisahkan train-test
    def trainTestSplit(self, file):
        df = self.fileAsDataframe(file, 'csv')
        train, test = self.wayOfSplitting(df)

        return train, test 
    
    # fungsi tokenisasi
    def tokenisasi(self, text):
        # inisialisasi token dan assign variable token
        self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        # tokenisasi manual teks
        input_ids = [bos_token_id] + self.tokenizer.encode(text) + [eos_token_id]

        return input_ids 


    # main fungsi tokenisasi
    def mainTokenise(self, singleInput='', multiInput=[]):
        inputs = ''
        input_ids = ''

        if singleInput:
            inputs = singleInput

            input_ids = self.tokenisasi(inputs)

        elif multiInput:
            inputs = multiInput
            input_ids = [self.tokenisasi(inp) for inp in inputs]

        else:
            raise 'Please give function single or multi input'
        
        return inputs, input_ids




        

