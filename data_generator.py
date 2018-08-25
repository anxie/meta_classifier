""" Code for loading data and generating data batches during training """
from __future__ import division

import copy
import logging
import os
import glob
import tempfile
import pickle
from datetime import datetime
from collections import OrderedDict

import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import flags
from natsort import natsorted
from random import shuffle
import h5py

FLAGS = flags.FLAGS


class DataGenerator(object):
    def __init__(self):
        self.update_batch_size = FLAGS.update_batch_size
        self.test_batch_size = FLAGS.train_update_batch_size if FLAGS.train_update_batch_size != -1 else self.update_batch_size
        self.meta_batch_size = FLAGS.meta_batch_size
        self.data_dir = FLAGS.data_dir

        self.num_tasks = range(FLAGS.num_tasks)
        split = int(FLAGS.train_val_split * FLAGS.num_tasks)
        self.train_idx = self.num_tasks[split:]
        self.val_idx = self.num_tasks[:split]

    def generate_batches(self):
        train_img_folders = {i: os.path.join(self.data_dir, 'object_%d' % i) for i in self.train_idx}
        val_img_folders = {i: os.path.join(self.data_dir, 'object_%d' % i) for i in self.val_idx}

        if FLAGS.train:
            TEST_PRINT_INTERVAL = 500
        else:
            TEST_PRINT_INTERVAL = 10
        TOTAL_ITERS = FLAGS.metatrain_iterations

        self.all_training_filenames = []
        self.all_training_labels = []
        self.all_val_filenames = []
        self.all_val_labels = []

        for itr in xrange(TOTAL_ITERS):
            sampled_train_idx = random.sample(self.train_idx, self.meta_batch_size)

            for idx in sampled_train_idx:
                sampled_folder = train_img_folders[idx]
                success_image_paths = natsorted([x for x in os.listdir(sampled_folder) if 'fail' not in x])
                fail_image_paths = natsorted([x for x in os.listdir(sampled_folder) if 'fail' in x])

                fail_subset_idx = np.random.choice(range(len(fail_image_paths)), size=len(success_image_paths), replace=False)
                fail_image_paths = [fail_image_paths[i] for i in fail_subset_idx]

                image_paths = success_image_paths + fail_image_paths

                sampled_success_idx = np.random.choice(range(len(success_image_paths)),
                                                       size=self.update_batch_size,
                                                       replace=False)

                sampled_success_image_paths = [image_paths[i] for i in sampled_success_idx]

                remaining_image_paths = [path for path in image_paths if path not in sampled_success_image_paths]

                sampled_either_idx = np.random.choice(range(len(remaining_image_paths)),
                                                      size=self.test_batch_size,
                                                      replace=False)

                sampled_success_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_success_idx]
                sampled_success_labels = [1 if 'fail' not in image else 0 for image in sampled_success_images]

                sampled_either_images = [os.path.join(sampled_folder, remaining_image_paths[i]) for i in sampled_either_idx]
                sampled_either_labels = [1 if 'fail' not in image else 0 for image in sampled_either_images]

                self.all_training_filenames.extend(sampled_success_images)
                self.all_training_filenames.extend(sampled_either_images)
                self.all_training_labels.extend(sampled_success_labels)
                self.all_training_labels.extend(sampled_either_labels)

            if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                sampled_val_idx = random.sample(self.val_idx, self.meta_batch_size)
                for idx in sampled_val_idx:
                    sampled_folder = val_img_folders[idx]
                    success_image_paths = natsorted([x for x in os.listdir(sampled_folder) if 'fail' not in x])
                    fail_image_paths = natsorted([x for x in os.listdir(sampled_folder) if 'fail' in x])

                    fail_subset_idx = np.random.choice(range(len(fail_image_paths)), size=len(success_image_paths), replace=False)
                    fail_image_paths = [fail_image_paths[i] for i in fail_subset_idx]

                    image_paths = success_image_paths + fail_image_paths

                    sampled_success_idx = np.random.choice(range(len(success_image_paths)),
                                                           size=self.update_batch_size,
                                                           replace=False)

                    sampled_success_image_paths = [image_paths[i] for i in sampled_success_idx]

                    remaining_image_paths = [path for path in image_paths if path not in sampled_success_image_paths]

                    sampled_either_idx = np.random.choice(range(len(remaining_image_paths)),
                                                          size=self.test_batch_size,
                                                          replace=False)

                    sampled_success_images = [os.path.join(sampled_folder, success_image_paths[i]) for i in sampled_success_idx]
                    sampled_success_labels = [1 if 'fail' not in image else 0 for image in sampled_success_images]

                    sampled_either_images = [os.path.join(sampled_folder, remaining_image_paths[i]) for i in sampled_either_idx]
                    sampled_either_labels = [1 if 'fail' not in image else 0 for image in sampled_either_images]

                    self.all_val_filenames.extend(sampled_success_images)
                    self.all_val_filenames.extend(sampled_either_images)
                    self.all_val_labels.extend(sampled_success_labels)
                    self.all_val_labels.extend(sampled_either_labels)

    def make_batch_tensor(self, restore_iter=0, train=True):
        TEST_INTERVAL = 500
        batch_image_size = (self.update_batch_size + self.test_batch_size) * self.meta_batch_size

        if train:
            all_filenames = self.all_training_filenames
            all_labels = self.all_training_labels
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size*(restore_iter+1):]
                all_labels = all_labels[batch_image_size*(restore_iter+1):]
        else:
            all_filenames = self.all_val_filenames
            all_labels = self.all_val_labels
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size*(int(restore_iter/TEST_INTERVAL)+1):]
                all_labels = all_labels[batch_image_size * (int(restore_iter / TEST_INTERVAL) + 1):]

        im_height = FLAGS.im_height
        im_width = FLAGS.im_width
        num_channels = FLAGS.num_channels

        images = tf.convert_to_tensor(all_filenames, dtype=tf.string)
        labels = tf.convert_to_tensor(all_labels, dtype=tf.uint8)

        input_queue = tf.train.slice_input_producer([images, labels], shuffle=False)

        print 'Generating image processing ops'
        file_content = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(file_content, channels=num_channels)
        image.set_shape((im_height, im_width, num_channels))
        image = tf.cast(image, tf.float32) / 255.0

        if FLAGS.aug and train:
            image_hsv = tf.image.rgb_to_hsv(image)
            image_h = image_hsv[:, :, 0]
            image_s = image_hsv[:, :, 1]
            image_v = image_hsv[:, :, 2]
            image_v = tf.clip_by_value(image_v * tf.random_uniform([1], 0.5, 1.5), 0.0, 1.0)
            image_hsv = tf.stack([image_h, image_s, image_v], 2)
            image = tf.image.hsv_to_rgb(image_hsv)

            image = tf.contrib.image.translate(image, tf.random_uniform([2], -8, 8))

        label = input_queue[1]

        num_preprocess_threads = 1
        min_queue_examples = 64

        print 'Batching images'
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )

        all_images, all_labels = [], []
        for i in xrange(self.meta_batch_size):
            images = image_batch[i*(self.update_batch_size+self.test_batch_size):(i+1)*(self.update_batch_size+self.test_batch_size)]
            images = tf.reshape(images, [self.update_batch_size+self.test_batch_size, FLAGS.im_height, FLAGS.im_width, FLAGS.num_channels])

            labels = label_batch[i*(self.update_batch_size+self.test_batch_size):(i+1)*(self.update_batch_size+self.test_batch_size)]
            labels = tf.reshape(labels, [self.update_batch_size+self.test_batch_size, -1])

            all_images.append(images)
            all_labels.append(labels)

        all_images = tf.stack(all_images)
        all_labels = tf.stack(all_labels)
        all_labels = tf.squeeze(tf.one_hot(all_labels, 2))
        return all_images, all_labels
