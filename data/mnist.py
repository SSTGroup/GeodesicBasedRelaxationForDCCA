import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import KFold


class MNISTData:
    def __init__(
        self,
        batch_size,
        num_batches,
        classes,
        num_classes,
        view1_train,
        view1_eval,
        view1_test,
        view2_train,
        view2_eval,
        view2_test,
        num_samples,
        dim_samples,
    ):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.classes = classes
        self.num_classes = num_classes
        self.view1_train = view1_train
        self.view1_eval = view1_eval
        self.view1_test = view1_test
        self.view2_train = view2_train
        self.view2_eval = view2_eval
        self.view2_test = view2_test
        self.num_samples = num_samples
        self.dim_samples = dim_samples

        self.training_data = self.get_dataset(view1_train, view2_train)
        self.eval_data = self.get_dataset(view1_eval, view2_eval)
        self.test_data = self.get_dataset(view1_test, view2_test)

    @classmethod
    def generate(
        cls,
        batch_size,
        num_boxes,
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        max_width=10,
        flatten=True,
        num_samples=50000,
    ):
        num_batches = int(np.floor(num_samples / batch_size))
        num_classes = len(classes)

        augment_ds = tfds.as_numpy(tfds.load("mnist_corrupted/spatter"))

        # View 1
        print("View 1:")
        print(" - Pre-process")
        view1_train, view1_eval, view1_test = preprocess_dataset(augment_ds, flatten=flatten)
        print(" - Filter classes")
        view1_train = filter_classes(view1_train, classes=classes)
        view1_eval = filter_classes(view1_eval, classes=classes)
        view1_test = filter_classes(view1_test, classes=classes)
        print(" - Add boxes")
        view1_train = augment_batch(view1_train, num_boxes=num_boxes, max_width=max_width, flatten=flatten)
        view1_eval = augment_batch(view1_eval, num_boxes=num_boxes, max_width=max_width, flatten=flatten)
        view1_test = augment_batch(view1_test, num_boxes=num_boxes, max_width=max_width, flatten=flatten)

        # View 2
        print("\nView 2:")
        print(" - Pre-process")
        view2_train, view2_eval, view2_test = preprocess_dataset(augment_ds, flatten=flatten)
        print(" - Filter classes")
        view2_train = filter_classes(view2_train, classes=classes)
        view2_eval = filter_classes(view2_eval, classes=classes)
        view2_test = filter_classes(view2_test, classes=classes)
        print(" - Add boxes")
        view2_train = augment_batch(view2_train, num_boxes=num_boxes, max_width=max_width, flatten=flatten)
        view2_eval = augment_batch(view2_eval, num_boxes=num_boxes, max_width=max_width, flatten=flatten)
        view2_test = augment_batch(view2_test, num_boxes=num_boxes, max_width=max_width, flatten=flatten)
        print(" - Shuffle")
        view2_train = shuffle_instances(view2_train)
        view2_eval = shuffle_instances(view2_eval)
        view2_test = shuffle_instances(view2_test)

        dim_samples = view1_train[0].shape[1]

        return cls(
            batch_size=batch_size,
            num_batches=num_batches,
            classes=classes,
            num_classes=num_classes,
            view1_train=(view1_train[0][:num_samples, :], view1_train[1][:num_samples]),
            view1_eval=view1_eval,
            view1_test=view1_test,
            view2_train=(view2_train[0][:num_samples, :], view2_train[1][:num_samples]),
            view2_eval=view2_eval,
            view2_test=view2_test,
            num_samples=num_samples,
            dim_samples=dim_samples,
        )

    @classmethod
    def from_saved_data(cls, save_path):
        with open(os.path.join(save_path, "data.pkl"), "rb") as f:
            save_dict = pkl.load(f)

        return cls(
            batch_size=save_dict["batch_size"],
            num_batches=save_dict["num_batches"],
            classes=save_dict["classes"],
            num_classes=save_dict["num_classes"],
            view1_train=save_dict["view1_train"],
            view1_eval=save_dict["view1_eval"],
            view1_test=save_dict["view1_test"],
            view2_train=save_dict["view2_train"],
            view2_eval=save_dict["view2_eval"],
            view2_test=save_dict["view2_test"],
            num_samples=save_dict["num_samples"],
            dim_samples=save_dict["dim_samples"],
        )

    def save(self, dir):
        save_dict = {
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "classes": self.classes,
            "num_classes": self.num_classes,
            "view1_train": self.view1_train,
            "view1_eval": self.view1_eval,
            "view1_test": self.view1_test,
            "view2_train": self.view2_train,
            "view2_eval": self.view2_eval,
            "view2_test": self.view2_test,
            "num_samples": self.num_samples,
            "dim_samples": self.dim_samples,
        }

        with open(os.path.join(dir, "data.pkl"), "wb") as f:
            pkl.dump(save_dict, f)

    def get_dataset(self, view1, view2):
        assert np.all(view1[1] == view2[1])
        # Create dataset from numpy array in dict
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "nn_input_0": view1[0],
                "nn_input_1": view2[0],
                "labels": view1[1],
            }
        )

        # Batch
        dataset = dataset.batch(self.batch_size)

        dataset = dataset.cache()

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def plot(self):
        index = np.random.choice(range(self.view1_train[0].shape[0]), size=3)

        fig, ax = plt.subplots(3, 2, figsize=(10, 10))

        for i in range(3):
            ax[i, 0].imshow(self.view1_train[0][index[i]].reshape((28, 28)), cmap="gray")
            ax[i, 0].set_title(str(self.view1_train[1][index[i]]))
            ax[i, 1].imshow(self.view2_train[0][index[i]].reshape((28, 28)), cmap="gray")
            ax[i, 1].set_title(str(self.view2_train[1][index[i]]))

    def plot_eval(self):
        index = np.random.choice(range(self.view1_eval[0].shape[0]), size=3)

        fig, ax = plt.subplots(3, 2, figsize=(10, 10))

        for i in range(3):
            ax[i, 0].imshow(self.view1_eval[0][index[i]].reshape((28, 28)), cmap="gray")
            ax[i, 0].set_title(str(self.view1_eval[1][index[i]]))
            ax[i, 1].imshow(self.view2_eval[0][index[i]].reshape((28, 28)), cmap="gray")
            ax[i, 1].set_title(str(self.view2_eval[1][index[i]]))


def preprocess_dataset(ds, flatten):
    images_train = np.zeros((60000, 28, 28, 1), dtype=np.uint8)
    labels_train = np.zeros((60000,), dtype=np.uint8)
    for i, example in enumerate(ds["train"]):
        images_train[i] = example["image"]
        labels_train[i] = example["label"]

    images_test = np.zeros((10000, 28, 28, 1), dtype=np.uint8)
    labels_test = np.zeros((10000,), dtype=np.uint8)
    for i, example in enumerate(ds["test"]):
        images_test[i] = example["image"]
        labels_test[i] = example["label"]

    images_eval = images_train[50000:] / 255
    if flatten:
        images_eval = images_eval.reshape((10000, -1))
    labels_eval = labels_train[50000:]

    images_train = images_train[:50000] / 255
    if flatten:
        images_train = images_train.reshape((50000, -1))
    labels_train = labels_train[:50000]

    images_test = images_test / 255
    if flatten:
        images_test = images_test.reshape((10000, -1))

    return (images_train, labels_train), (images_eval, labels_eval), (images_test, labels_test)


def add_noise_to_dataset(set_train, set_eval, set_test):
    noisy_images_train = add_noise_to_images(set_train[0])

    noisy_images_eval = add_noise_to_images(set_eval[0])

    noisy_images_test = add_noise_to_images(set_test[0])

    return (noisy_images_train, set_train[1]), (noisy_images_eval, set_eval[1]), (noisy_images_test, set_test[1])


def add_noise_to_images(images):
    noise = np.random.uniform(low=0.0, high=0.5, size=images.shape)
    images += noise
    images[images > 1] = 1
    return images


def shuffle_instances(instances_set):
    images = instances_set[0]
    labels = instances_set[1]

    shuffled_images = np.zeros(images.shape)
    for i in range(shuffled_images.shape[0]):
        shuffled_images[i] = images[np.random.choice(np.where(labels == labels[i])[0])]

    return (shuffled_images, labels)


def box_augment(image, num_boxes, max_width):
    mask = np.zeros_like(image)
    size_x, size_y = mask.shape[:2]

    for i in range(num_boxes):
        # Make white box
        widthx, widthy = np.random.randint(3, max_width, size=2)
        box = np.array([1] * widthx * widthy).reshape(widthx, widthy)
        # Sample center and width
        x, y = np.random.randint(0, size_x - widthx), np.random.randint(0, size_y - widthy)
        mask[x : x + widthx, y : y + widthy, 0] = box

    augm_image = image + mask
    augm_image[augm_image > 1] = 1

    return augm_image


def augment_slice_batch(view, num_boxes, max_width, flatten):
    images = view[0].reshape((-1, 8, 24, 1))
    augm_images = np.zeros_like(images)
    for i in range(images.shape[0]):
        augm_images[i, :] = box_augment(images[i, :], num_boxes=num_boxes, max_width=max_width)
    if flatten:
        augm_images = augm_images.reshape(-1, 8 * 24)

    return (augm_images, view[1])


def augment_batch(view, num_boxes, max_width, flatten):
    images = view[0].reshape((-1, 28, 28, 1))
    augm_images = np.zeros_like(images)
    for i in range(images.shape[0]):
        augm_images[i, :] = box_augment(images[i, :], num_boxes=num_boxes, max_width=max_width)
    if flatten:
        augm_images = augm_images.reshape(-1, 28 * 28)

    return (augm_images, view[1])


def filter_classes(view, classes):
    images, labels = view
    ids = np.zeros_like(labels, dtype=bool)
    for cl in classes:
        ids = np.logical_or(ids, labels == cl)

    return (images[ids], labels[ids])
