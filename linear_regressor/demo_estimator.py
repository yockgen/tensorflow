from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import demo_data02 as data
import datetime
import re
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--training_path',required=True, help='training source path')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--train_steps', default=500, type=int, help='number of training steps')


def reject_outliers(data, m = 0.02):

    medianPoint =np.median(data)
    d = np.abs(data - medianPoint)

    result = []
    for idx,itx in enumerate(d):
        if itx > m:
            nTmp = str(idx+1).zfill(2) 
            result.append(nTmp)
        
    
    return result

def main(argv):

    
    
    aa = datetime.datetime.now().replace(microsecond=0)
    args = parser.parse_args(argv[1:])

    # Fetch the data
    source_file = args.training_path
    (train_sum,train_x, train_y), (test_x, test_y) = data.load_data(source_file)

    
    
       
    
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for idx in train_x.keys():            
        categorical_column_tmp = tf.feature_column.categorical_column_with_hash_bucket(key=idx, hash_bucket_size=1000)
        my_feature_columns.append(tf.feature_column.indicator_column(categorical_column_tmp))
        
    
    #classifier = tf.estimator.LinearRegressor(
    classifier = tf.estimator.LinearClassifier(    
        feature_columns=my_feature_columns,                
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

    
    #'PARAM01','PARAM02','PARAM03','PARAM04','PARAM05','PARAM06','PARAM07','RESULT'
    #99% 'POSEDGE''-5V''PIN01''TEST08.SOF''6337''/TESTPATH/PR14''3274'
    #'POSEDGE','-5V','PIN04','TEST01.SOF','1436','/TESTPATH/PR14','5841'

    #predict_x = {
     #   'PARAM01': ["'POSEDGE'"],
      #  'PARAM02': ["'-5V'"],
       # 'PARAM03': ["'PIN04'"],
        #'PARAM04': ["'TEST01.SOF'"],
        #'PARAM05': ["'1436'"],
        #'PARAM06': ["'/TESTPATH/PR14'"],
        #'PARAM07': ["'5841'"]      
    #}

        
    #predictions = classifier.predict(input_fn=lambda:data.eval_input_fn(predict_x,labels=None, batch_size=args.batch_size))
    #for pred_dict in predictions:
     #   print (pred_dict)



    #print (tf.train.list_variables('./tmp/demo'))    #list all available variables
    sumWeight = []
    orgWeight = []

    #print (classifier.get_variable_value('linear/linear_model/SEX_indicator/weights').flatten())    
    for i in data.CSV_COLUMN_NAMES[:-1]:
        sIdx = i
        keyval = 'linear/linear_model/' + sIdx + '_indicator/weights'
        paramweights = np.sum(classifier.get_variable_value('linear/linear_model/' + sIdx + '_indicator/weights').flatten())
        orgWeight.append(paramweights)
        sumWeight.append(abs(paramweights))
        print ("param %s = %s" % (sIdx,paramweights))


    
    
    #outliers = reject_outliers(sumWeight)
    #print ("outlier=",outliers)

    #predict_x = {
     #   'CLASS': ["CLASS1","CLASS3","CLASS3"],
      #  'SEX': ["adults","adults","adults"],
       # 'AGE': ["women","man","women"]        
    #}

    #predict_x = {
     #   'LEFT': ["GOOD","GOOD","BAD"],
      #  'RIGHT': ["GOOD","BAD","BAD"],
       # 'DUMMY': ["NIL","NIL","NIL"]        
    #}      
    
        
    #predictions = classifier.predict(input_fn=lambda:data.eval_input_fn(predict_x,labels=None, batch_size=args.batch_size))
    #for pred_dict in predictions:
     #   print (pred_dict)

    




    
    #distinct all conditions
    predict_x = train_x.drop_duplicates()    


    predictions = classifier.predict(
        input_fn=lambda:data.eval_input_fn(predict_x,
                                                labels=None, batch_size=args.batch_size))


    template = ('\nPrediction is "{}" ({:.1f}%)')
    for pred_dict,row in zip(predictions,predict_x.iterrows()):
        #print (row)
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]                

        cols = ""
        colsdisplay=""
        for itx in row[1]:
            cols = cols + itx
            colsdisplay= colsdisplay + itx + ","

        cols = cols + str(class_id) 

        count = 0
        for itx in train_sum:
            temp = itx[0][0]
            temp01 = ""
            for y in temp:                
                temp01 = temp01 + str(y)
            
            if (temp01==cols):
                count = itx[1]
            
        print ("%s **Result = %s (confidence=%s percent), cases=%s" % (colsdisplay, class_id,100*probability,count))
    

    #show chart
    objects = data.CSV_COLUMN_NAMES[:-1] #('PARAM01','PARAM02','PARAM03','PARAM04','PARAM05','PARAM06','PARAM07')
    y_pos = np.arange(len(sumWeight))
     
    plt.bar(y_pos, sumWeight, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Weight')
    plt.title('Parameter Weight Score')
     
    plt.show()
    


    bb = datetime.datetime.now().replace(microsecond=0)

    print("\n\ntime=",bb-aa)

    
if __name__ == '__main__':    

    
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

    
    
