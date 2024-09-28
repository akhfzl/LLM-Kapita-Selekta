from utils.dataPreprocess import DataPreprocess


if __name__ == '__main__':
    variable_object = DataPreprocess('corpus/json', 'new_dataset')
    # files = variable_object.openFolder()
    
    # split small and large corpus for prepare
    # loader_df = variable_object.readAllData(files)
    # print(loader_df)
    # devided_small =  round(len(loader_df) * 0.2)
    # df_small, df_large = loader_df[(len(loader_df) - devided_small): ], loader_df[:(len(loader_df) - devided_small)]
    # df_small.to_csv('small_corpus.csv', index=False)
    # df_large.to_csv('large_corpus.csv', index=False)

    # load data preprocess

    df = variable_object.mainTokenise(multiInput=[
        "Ini adalah contoh teks pertama.",
        "Ini contoh teks kedua untuk tokenisasi lebih lanjut."
    ])
    print(df)
    
