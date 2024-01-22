from __future__ import division
from __future__ import print_function
from sklearn.cluster import KMeans
import os

# # Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# import tensorflow as tf
import tensorflow.compat.v1 as tf
# from metrics import clustering_metrics
from constructor import get_placeholder, get_model, format_data, get_optimizer, update
from tqdm import tqdm
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

class Clustering_Runner():
    def __init__(self, settings):

        print("Clustering on dataset: %s, model: %s, number of iteration: %3d" % (settings['data_name'], settings['model'], settings['iterations']))

        self.data_name = settings['data_name']
        self.iteration =settings['iterations']
        self.model = settings['model']
        self.n_clusters = settings['clustering_num']

    def erun(self, seed):
        model_str = self.model

        # formatted data
        feas = format_data(self.data_name)

        # Define placeholders
        placeholders = get_placeholder(feas['adj'])

        # construct model
        d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

        # Initialize session
        sess = tf.Session()
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in tqdm(range(self.iteration), total=self.iteration):
            emb, loss = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])

            # if (epoch+1) % 2 == 0:
            #     kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(emb)
            #     print("Epoch:", '%04d' % (epoch + 1))
            #     predict_labels = kmeans.predict(emb)
            #     cm = clustering_metrics(feas['true_labels'], predict_labels)
            #     cm.evaluationClusterModelFromLabel()

            if epoch % 10 == 0:
                tqdm.write("Epoch {}/{} Loss: {}".format(epoch+1, self.iteration, loss))

        emb, _ = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
        import numpy as np
        import os
        os.makedirs("outputs", exist_ok=True)
        if self.model == "arga_ae":
            np.savez("outputs/ARGA_{}_emb_{}.npz".format(self.data_name, seed), emb=emb)
        elif self.model == "arga_vae":
            np.savez("outputs/ARVGA_{}_emb_{}.npz".format(self.data_name, seed), emb=emb)