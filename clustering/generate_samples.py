import tensorflow as tf
import numpy as np

from helper import *

n_features = 2
n_clusters = 3
n_samples_per_cluster = 300
seed = 700
embiggen_factor = 70



data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)


model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)

    actual_centroid_values = session.run(data_centroids) #for benchmarking   
    sample_values = session.run(samples)




    updated_centroid_value = session.run(updated_centroids)
    prev_value = session.run(tf.reduce_sum(tf.abs(updated_centroid_value)))
    

    for i in range(500):

        updated_centroid_value = session.run(updated_centroids)

        #current_value = session.run(tf.reduce_sum(tf.abs(updated_centroid_value)))
        current_value = session.run(tf.reduce_mean(updated_centroid_value))

        delta = abs(prev_value - current_value)
        print ("%s prev_value=%s,current_value=%s %s\n" % (i,prev_value,current_value,delta))
        if delta <0.06:
            print "unsupervised"
            break;

        #prev_value = session.run(tf.reduce_sum(tf.abs(updated_centroid_value)))
        prev_value = session.run(tf.reduce_mean(updated_centroid_value))


        
        #loss = session.run(tf.subtract(updated_centroid_value, actual_centroid_values))        
        #loss_sum = session.run(tf.reduce_sum(tf.abs(loss)))        
        #print loss_sum
        #print "\n"
        #if loss_sum <= 5: break
        
    

    print("\n\ncentroid value=\n%s\n\n" % updated_centroid_value)
    print("\n\nactual centroid value=\n%s\n\n" % actual_centroid_values)

    

plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)
