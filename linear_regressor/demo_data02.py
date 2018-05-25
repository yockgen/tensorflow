import pandas as pd
import tensorflow as tf
import numpy as np
import re





#CSV_COLUMN_NAMES = ['PARAM01','PARAM02','PARAM03','PARAM04','PARAM05','PARAM06','PARAM07','RESULT']
CSV_COLUMN_NAMES = []



def load_data(train_path,y_name='LABEL'):
    

    #get column name from first line
    header = pd.read_csv(train_path,header=None,nrows=1)    
    for index, row in header.iterrows():    
        for col in row:            
           CSV_COLUMN_NAMES.append(col)

    
    
    
    #read actual data
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)    
    

    #select distinct and count occurence of each case (select count(*) from data group by *)
    train_cat = list(train.groupby(CSV_COLUMN_NAMES))    
    train_count= list(train.groupby(CSV_COLUMN_NAMES).size())
    train_sum= zip (train_cat,train_count)
    

    # use half training data as test data
    random_indices = np.random.choice(len(train.index.values)/2,4,replace=False) #get half for test
    test=train.loc[random_indices]

     
    # return result 
    train_x, train_y = train, train.pop(y_name)
    test_x, test_y = test, test.pop(y_name)

    return (train_sum,train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


