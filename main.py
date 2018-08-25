import numpy as np
import random
import tensorflow as tf

import matplotlib.pyplot as plt

from data_generator import DataGenerator
from msc import MSC
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('data_dir', '/home/annie/Desktop/meta_classifier/data', 'directory of data')
flags.DEFINE_integer('num_tasks', 92, 'number of tasks in dataset')
flags.DEFINE_float('train_val_split', 0.1, 'train/validation set split')
flags.DEFINE_integer('im_height', 56, 'height of images')
flags.DEFINE_integer('im_width', 64, 'width of images')
flags.DEFINE_integer('num_channels', 3, 'number of channels in input image')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 1000, 'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 8, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1.0, 'step size alpha for inner gradient update.')
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_bool('grad_clip', True, 'use gradient clipping')
flags.DEFINE_float('clip_min', -80.0, 'minimum for gradient clipping')
flags.DEFINE_float('clip_max', 80.0, 'maximum for gradient clipping')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_bool('aug', True, 'use data augmentation')

## Model options
flags.DEFINE_string('norm', 'layer_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_conv_layers', 3, 'number of convolutional layers')
flags.DEFINE_integer('num_filters', 32, 'number of filters for conv nets.')
flags.DEFINE_integer('num_fc_layers', 2, 'number of fully connected layers')
flags.DEFINE_integer('hidden_dim', 40, 'hidden dimension of fully connected layers')
flags.DEFINE_bool('fp', True, 'use feature spatial soft-argmax')
flags.DEFINE_string('vgg_path', '/home/annie/Desktop/meta_classifier/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'path to weights for first layer of VGG')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_string('model_file', '/home/annie/Desktop/meta_classifier/pretrained_model/model49999', 'path to the pretrained model')

flags.DEFINE_bool('train', False, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', 5, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step during training. (use if you want to test with a different value)') # 0.1 for omniglot


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    lossesa, postlosses = [], []
    accsa, postaccs = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors)

        if itr % SUMMARY_INTERVAL == 0:
            lossesa.append(result[-4])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-3])
            accsa.append(result[-2])
            postaccs.append(result[-1])

        if itr != 0 and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(lossesa)) + ' ' + str(np.mean(accsa)) + ', ' \
                         + str(np.mean(postlosses)) + ' ' + str(np.mean(postaccs))
            print(print_str)
            lossesa, postlosses = [], []
            accsa, postaccs = [], []

        if itr != 0 and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
            feed_dict = {}
            input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1],
                             model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1],
                             model.summ_op]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: '
                  + str(result[0]) + ' ' + str(result[2]) + ', ' \
                  + str(result[1]) + ' ' + str(result[3]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


def test(model, saver, sess, num_test_points):
    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(num_test_points):
        feed_dict = {model.meta_lr : 0.0}

        result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        metaval_accuracies.append(result[-1])

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(num_test_points)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))


def main():
    if FLAGS.train:
        test_num_updates = 1  # eval on at least one update during training
    else:
        test_num_updates = 10

    if not FLAGS.train:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # FLAGS.meta_batch_size = 1

    data_generator = DataGenerator()
    data_generator.generate_batches()

    if FLAGS.train:
        tf.set_random_seed(5)
        np.random.seed(5)
        random.seed(5)
        image_tensor, label_tensor = data_generator.make_batch_tensor()
        inputa = tf.slice(image_tensor, [0, 0, 0, 0, 0], [-1, FLAGS.update_batch_size, -1, -1, -1])
        inputb = tf.slice(image_tensor, [0, FLAGS.update_batch_size, 0, 0, 0], [-1, -1, -1, -1, -1])
        labela = tf.slice(label_tensor, [0, 0, 0], [-1, FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0, FLAGS.update_batch_size, 0], [-1, -1, -1])
        input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    tf.set_random_seed(6)
    np.random.seed(6)
    random.seed(6)
    image_tensor, label_tensor = data_generator.make_batch_tensor(train=False)
    inputa = tf.slice(image_tensor, [0, 0, 0, 0, 0], [-1, FLAGS.update_batch_size, -1, -1, -1])
    inputb = tf.slice(image_tensor, [0, FLAGS.update_batch_size, 0, 0, 0], [-1, -1, -1, -1, -1])
    labela = tf.slice(label_tensor, [0, 0, 0], [-1, FLAGS.update_batch_size, -1])
    labelb = tf.slice(label_tensor, [0, FLAGS.update_batch_size, 0], [-1, -1, -1])
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    model = MSC(test_num_updates=test_num_updates)
    if FLAGS.train:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if FLAGS.vgg_path:
        var_list.extend([model.weights['conv1_w'], model.weights['conv1_b']])
    saver = tf.train.Saver(var_list, max_to_keep=10)

    sess = tf.InteractiveSession()

    if not FLAGS.train:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'mbs' + str(FLAGS.meta_batch_size) + '.ubs' + str(FLAGS.update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr) + \
                    '.metalr' + str(FLAGS.meta_lr) + '.conv' + str(FLAGS.num_conv_layers) + '.filters' + str(FLAGS.num_filters) + '.fc' + str(FLAGS.num_fc_layers) + '.dim' + str(FLAGS.hidden_dim) + '.' + str(FLAGS.norm)

    if FLAGS.fp:
        exp_string += '.fp'

    resume_itr = 0

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.model_file:
        saver.restore(sess, FLAGS.model_file)
    elif FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, num_test_points=100)

if __name__ == "__main__":
    main()
