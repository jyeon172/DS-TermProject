import pandas as pd
import numpy as np

def one_hot(dt, data_idx, prefix):  # one-hot encoding
    all_ele = []
    data_col = dt.iloc[:, data_idx]  # get data column using index

    for i in data_col:  # get all elements of data
        all_ele.extend(i.split(','))

    ele = pd.unique(all_ele)  # make data elemtns unique
    zero_matrix = np.zeros((len(data_col), len(ele)))  # make zero table for one-hot
    dumnie = pd.DataFrame(zero_matrix, columns=ele)

    for i, elem in enumerate(data_col):  # update one-hot table 1 for each element
        index = dumnie.columns.get_indexer(elem.split(','))
        dumnie.iloc[i, index] = 1

    dt = dt.iloc[:, data_idx:]  # drop data before encoding
    data_joined = dt.join(dumnie.add_prefix(prefix))  # join one-hot encoding dataframe

    print('One-hot Encoding Success')
    return data_joined
