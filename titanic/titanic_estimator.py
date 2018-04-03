
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import titanic_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = titanic_data.load_data()

    print ("\ndataset=")
    print(train_x.describe())
    print ("\n")

    

    
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2,
        model_dir='./tmp/titanic',
        config=tf.contrib.learn.RunConfig(save_checkpoints_steps=250,save_checkpoints_secs=None,save_summary_steps=500,)
        )

    # Train the Model.
    classifier.train(
        input_fn=lambda:titanic_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:titanic_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


    # Generate predictions from the model
    expected = ['gone', 'survive']
   
    
    predict_x = train_x.drop_duplicates()
    print (predict_x)
    
    

    predictions = classifier.predict(
        input_fn=lambda:titanic_data.eval_input_fn(predict_x,
                                                labels=None, batch_size=args.batch_size))

    AGE = ['CHILD','ADULT']
    SEX = ['FEMALE','MALE'] 
    template = ('\nPrediction is "{}" ({:.1f}%)')
    for pred_dict,row in zip(predictions,predict_x.iterrows()):
        
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]        
        #print(template.format(titanic_data.SPECIES[class_id], 100 * probability))

        print ("Class=%s,Age=%s,Sex=%s, **Result = %s (confidence=%s percent)" % (row[1]['Class'],AGE[row[1]['Age']],SEX[row[1]['Sex']],titanic_data.SPECIES[class_id],100*probability))
        
        

    #predict_x = {
     #   'Class': [2, 1],
      #  'Age': [1, 0],
       # 'Sex': [1, 0]        
    #}
    #for pred_dict, expec in zip(predictions, expected):
        #template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        #class_id = pred_dict['class_ids'][0]
        #probability = pred_dict['probabilities'][class_id]

        
        #print(template.format(titanic_data.SPECIES[class_id],
         #                     100 * probability, expec))
        #print (pred_dict['probabilities'] * 100)

    print (tf.train.list_variables('./tmp/titanic'))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
