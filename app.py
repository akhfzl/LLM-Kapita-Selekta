from utils.dataModeling import DataModeling
import os

if not os.path.exists('/kaggle/working/small') or not os.path.exists('/kaggle/working/large'):
    os.mkdir('/kaggle/working/small')
    os.mkdir('/kaggle/working/large')

# small data
if __name__ == '__main__':  
    variable_object = DataModeling('corpus/json', '/kaggle/input/llm-pretrained/new_dataset')
    
    # split small and large corpus for prepare
    # files = variable_object.openFolder()
    # loader_df = variable_object.readAllData(files)
    # devided_small =  round(len(loader_df) * 0.2)
    # df_small, df_large = loader_df[(len(loader_df) - devided_small): ], loader_df[:(len(loader_df) - devided_small)]
    # df_small.to_csv('small_corpus.csv', index=False)
    # df_large.to_csv('large_corpus.csv', index=False)

    # load data preprocess small dataset
    train_small, val_small = variable_object.trainTestSplit(False)
    
    train_tokenisasi_small = variable_object.mainTokenise(list(train_small))
    val_tokenisasi_small = variable_object.mainTokenise(list(val_small))
    
    trainer_small = variable_object.TrainingModel(output_dir='/kaggle/working/small',train_dataset=train_tokenisasi_small, val_dataset=val_tokenisasi_small)
    evaluate_small = trainer_small.evaluate()

# large data
if __name__ == '__main__':  
    variable_object = DataModeling('corpus/json', '/kaggle/input/llm-pretrained/new_dataset')
    
    # split small and large corpus for prepare
    # files = variable_object.openFolder()
    # loader_df = variable_object.readAllData(files)
    # devided_small =  round(len(loader_df) * 0.2)
    # df_small, df_large = loader_df[(len(loader_df) - devided_small): ], loader_df[:(len(loader_df) - devided_small)]
    # df_small.to_csv('small_corpus.csv', index=False)
    # df_large.to_csv('large_corpus.csv', index=False)

    # load data preprocess small dataset
    train_large, val_large = variable_object.trainTestSplit(True)
    
    train_tokenisasi_large = variable_object.mainTokenise(list(train_large))
    val_tokenisasi_large = variable_object.mainTokenise(list(val_large))
    
    trainer_large = variable_object.TrainingModel(output_dir='/kaggle/working/large',train_dataset=train_tokenisasi_large, val_dataset=val_tokenisasi_large)
    evaluate_large = trainer_large.evaluate()
    
