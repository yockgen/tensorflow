from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import demo_data as data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--train_steps', default=100, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = data.load_data()

   
    print ("\ndataset=")
    print(train_x.describe())
    print ("\n")

    
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for idx in train_x.keys():    
        print (idx)
        categorical_column_tmp = tf.feature_column.categorical_column_with_hash_bucket(key=idx, hash_bucket_size=1000)
        my_feature_columns.append(tf.feature_column.indicator_column(categorical_column_tmp))
        
        

    #exit(1)
    #categorical_column1 = tf.feature_column.categorical_column_with_vocabulary_list(key="left", vocabulary_list=["BAD", "GOOD"], default_value=0)
    #categorical_column2 = tf.feature_column.categorical_column_with_vocabulary_list(key="right", vocabulary_list=["BAD", "GOOD"], default_value=0)
    #categorical_column1 = tf.feature_column.categorical_column_with_hash_bucket(key="left", hash_bucket_size=1000)
    #categorical_column2 = tf.feature_column.categorical_column_with_hash_bucket(key="right", hash_bucket_size=1000)

    
    #my_feature_columns = []    
    #my_feature_columns.append(tf.feature_column.indicator_column(categorical_column1))
    #my_feature_columns.append(tf.feature_column.indicator_column(categorical_column2))

      

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    #classifier = tf.estimator.LinearRegressor(
     #   feature_columns=my_feature_columns,                
      #  model_dir='./tmp/demo',
       # config=tf.contrib.learn.RunConfig(save_checkpoints_steps=5,save_checkpoints_secs=None,save_summary_steps=5,)
        #)

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2,
        model_dir='./tmp/demo',
        config=tf.contrib.learn.RunConfig(save_checkpoints_steps=250,save_checkpoints_secs=None,save_summary_steps=500,)
        )

    
    # Train the Model.
    classifier.train(
        input_fn=lambda:data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:data.eval_input_fn(test_x,test_y,batch_size=args.batch_size))

    print (eval_result)


    #GENERATE PREDICTION

    #distinct all conditions
    predict_x = train_x.drop_duplicates()
    
    
    
    predictions = classifier.predict(
        input_fn=lambda:data.eval_input_fn(predict_x,
                                                labels=None, batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%)')
    for pred_dict,row in zip(predictions,predict_x.iterrows()):
        
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]                

        print ("Left=%s,Right=%s, **Result = %s (confidence=%s percent)" % (row[1]['left'],row[1]['right'], class_id,100*probability))
    

    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
