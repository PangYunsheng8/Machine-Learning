import numpy as np
import pandas as pd

def get_data(train_path, test_path):
    """
    load train and test data
    """
    columns = [
    'age', 
    'workclass', 
    'fnlwgt', 
    'education', 
    'education-num', 
    'marital-status', 
    'occupation', 
    'relationship', 
    'race', 
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'Income']

    df_train = pd.read_csv(train_path, names=columns)
    df_test = pd.read_csv(test_path, names=columns, skiprows=1)
    return df_train, df_test

def preprocess_data(train_set, test_set):
    # 去掉包含缺失值的数据
    train_set = train_set.replace(' ?', np.nan).dropna()
    test_set = test_set.replace(' ?', np.nan).dropna()

    # 因为有受教育的年数，所以这里不需要教育这一列
    train_set.drop(["Education"], axis=1, inplace=True)
    test_set.drop(["Education"], axis=1, inplace=True)
    

if __name__ == '__main__':
    df_train, df_test = get_data(train_path='data/adult.data', test_path='data/adult.test')
    preprocess_data(df_train, df_test)
