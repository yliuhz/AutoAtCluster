# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import settings as stts

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner

import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf

import setproctitle

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("dataset", "cora", "")
    flags.DEFINE_string("model", "ARGA", "")
    flags.DEFINE_integer("nexp", 10, "")

    dataname = FLAGS.dataset       # 'cora' or 'citeseer' or 'pubmed'
    if FLAGS.model == "ARGA":
        model = 'arga_ae'          # 'arga_ae' or 'arga_vae'
    elif FLAGS.model == "ARVGA":
        model = "arga_vae"
    else:
        raise NotImplementedError()
    task = 'clustering'         # 'clustering' or 'link_prediction'


    flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
    flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
    # flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
    flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
    flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')

    # if dataname in ["cora", "citeseer", "wiki"]:
    flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('iterations', 200, 'number of iterations.')
    # elif dataname in ["pubmed"]:
    #     # flags.DEFINE_float('discriminator_learning_rate', 0.008, 'Initial learning rate.')
    #     flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
    #     # flags.DEFINE_integer('iterations', 2000, 'number of iterations.')
    #     flags.DEFINE_integer('iterations', 200, 'number of iterations.')

    seeds = np.arange(0, FLAGS.nexp, dtype=int)
    for seed in tqdm(seeds, total=FLAGS.nexp):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        setproctitle.setproctitle("{}-{}-{}".format(FLAGS.model, FLAGS.dataset[:2], seed))

        settings = stts.get_settings(dataname, model, task)

        if task == 'clustering':
            runner = Clustering_Runner(settings)
        else:
            runner = Link_pred_Runner(settings)

        runner.erun(seed)

