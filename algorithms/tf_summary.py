import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class TensorboardWriter():
    def __init__(self, root_dir):
        folders = list()
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if os.path.isdir(file_path):
                folders.append(file_path)

        curr_number = 0
        while True:
            num_str = str(curr_number)
            if len(num_str) == 1:
                num_str = "0"+num_str

            folder = os.path.join(root_dir, num_str)
            if not os.path.exists(folder):
                break
            else:
                curr_number = curr_number + 1

        os.makedirs(folder)

        self.writer = tf.summary.create_file_writer(folder)
        self.dir = folder

    def write_scalar_summary(self, epoch, list_of_tuples):
        with self.writer.as_default():
            for tup in list_of_tuples:
                tf.summary.scalar(tup[1], tup[0], step=epoch)
        self.writer.flush()
