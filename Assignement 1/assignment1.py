# Udacity deep learning
# Nicolas Nitche - 2016


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import hashlib

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

# A few global variables

num_classes = 10
np.random.seed(150)
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
display_pb2 = False
display_pb3 = False
display_pb4 = False
display_pb5 = False

# --------------------------------------------------------------------
# ---------- Downlaod the dataset if necessary  ----------------------
# --------------------------------------------------------------------

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

# --------------------------------------------------------------------
# ----------- Extract data if necessary ---------------------
# --------------------------------------------------------------------

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) -
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


if display_pb2:

    fig = plt.figure()
    fig.canvas.set_window_title('Pb 2: sample training set')
    for k in range(num_classes):
        current_data = pickle.load(open(train_datasets[k],'rb'))
        index = np.random.randint(0, current_data.shape[0])
        plt.subplot(2,5,(k+1))
        plt.imshow(current_data[index, :, :])
        plt.title(str(train_datasets[k]))


    fig = plt.figure()
    fig.canvas.set_window_title('Pb 2: sample test set')
    for k in range(num_classes):
        current_data = pickle.load(open(test_datasets[k],'rb'))
        index = np.random.randint(0, current_data.shape[0])
        plt.subplot(2,5,(k+1))
        plt.imshow(current_data[index, :, :])
        plt.title(str(test_datasets[k]))

    plt.show()

if display_pb3:

    fig = plt.figure()
    fig.canvas.set_window_title('Pb 3')
    ndata_per_class_train = np.zeros(num_classes)
    ndata_per_class_test = np.zeros(num_classes)

    for k in range(num_classes):
        ndata_per_class_train[k] = pickle.load(open(train_datasets[k],'rb')).shape[0]
        ndata_per_class_test[k] = pickle.load(open(test_datasets[k],'rb')).shape[0]

    plt.bar(np.arange(num_classes), ndata_per_class_train, width=0.5, color='b')
    plt.bar(np.arange(num_classes)+0.5, ndata_per_class_test, color='g',
            width=0.5 , tick_label = ['A','B','C','D','E','F','G','H','I','J'])
    plt.legend(('Train','Test'))
    plt.title('Number of samples per class')
    plt.show()

# --------------------------------------------------------------------
# Adapt data into array and create training, testing and validation
# dataset
# --------------------------------------------------------------------

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  return valid_dataset, valid_labels, train_dataset, train_labels

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

if display_pb4:

    fig = plt.figure()
    fig.canvas.set_window_title('Pb 4')
    ndata_per_class_train = np.zeros(num_classes)
    ndata_per_class_test = np.zeros(num_classes)
    ndata_per_class_valid = np.zeros(num_classes)

    for k in range(num_classes):
        ndata_per_class_train[k] = np.sum(train_labels==k)
        ndata_per_class_test[k] = np.sum(test_labels==k)
        ndata_per_class_valid[k] = np.sum(valid_labels==k)

    plt.bar(np.arange(num_classes), ndata_per_class_train, width=0.3, color='b')
    plt.bar(np.arange(num_classes)+0.3, ndata_per_class_test, width=0.3, color='g')
    plt.bar(np.arange(num_classes)+0.6, ndata_per_class_valid, color='r',
            width=0.3 , tick_label = ['A','B','C','D','E','F','G','H','I','J'])
    plt.legend(('Train','Test','Test'))
    plt.title('Number of samples per class after randomization')

    fig = plt.figure()
    fig.canvas.set_window_title('Pb 4 (bis)')

    index = np.random.randint(0,500)
    plt.subplot(1,3,1)
    plt.imshow(train_dataset[index, :, :])
    plt.title('Train sample /n for class ' + str(train_labels[index]))
    plt.subplot(1,3,2)
    plt.imshow(test_dataset[index, :, :])
    plt.title('Test sample /n for class ' + str(test_labels[index]))
    plt.subplot(1,3,3)
    plt.imshow(valid_dataset[index, :, :])
    plt.title('Test sample /n for class ' + str(valid_labels[index]))
    plt.show()

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


# measuring overlap between samples.

f = pickle.load(open(pickle_file, 'rb'))
train_dataset = f['train_dataset']
train_labels = f['train_labels']
valid_dataset = f['valid_dataset']
valid_labels = f['valid_labels']

def count_overlapping_element(list1, list2):
    count = 0
    for list1_element in list1:
        if list1_element in list2:
            count+=1

    return count

def create_hash_from_dataset(dataset):
    return np.array([hashlib.sha256(d).hexdigest() for d in dataset])

def find_overlap_hash(hash_d1, hash_d2):

    # fast way to find overlap
    overlap = {}
    for i, hash1 in enumerate(hash_d1):
        overlap_i = np.where(hash1==hash_d2)[0]
        if len(overlap_i):
            overlap[i] = overlap_i

    return overlap

import random
hash_training = create_hash_from_dataset(train_dataset)
hash_testing = create_hash_from_dataset(test_dataset)
hash_valid = create_hash_from_dataset(valid_dataset)

if display_pb5:

    hash_training = create_hash_from_dataset(train_dataset)
    hash_testing = create_hash_from_dataset(test_dataset)
    hash_valid = create_hash_from_dataset(valid_dataset)

    plt.figure() # training vs testing
    overlap = find_overlap_hash(hash_training, hash_testing)
    selected_key = random.sample(overlap.keys(), 1)[0]
    print(selected_key)
    plt.subplot(1,2,1)
    plt.imshow(train_dataset[selected_key,:,:])
    plt.title("Example of similar elements")
    plt.subplot(1,2,2)
    plt.imshow(test_dataset[overlap[selected_key][0],:,:])
    plt.title("There is %s elements /n of training set in testing dataset" % count_overlapping_element(hash_training, hash_testing))

    plt.show()
# training vs validation
print("There is %s elements /n of training set in validation dataset" % count_overlapping_element(hash_training, hash_valid))
# testing vs validation
print("There is %s elements /n of testing set in validation dataset" % count_overlapping_element(hash_testing, hash_valid))


# Dataset cleaning

pickle_sfile = 'satinitized_notMNIST.pickle'

if not os.path.isdir(pickle_sfile):

    # find elements that are unique to training/validation/test dataset
    overlap_train_test = find_overlap_hash(hash_training, hash_testing)
    overlap_train_valid = find_overlap_hash(hash_training, hash_valid)

    # In training test, remove all elements that are common to test set
    non_overlapping_idx = list(set(range(train_dataset.shape[0])) - set(overlap_train_test.keys()) - set(overlap_train_valid.keys()))
    sanitized_train_dataset = train_dataset[non_overlapping_idx]
    sanitized_train_label = train_labels[non_overlapping_idx]


    try:
      f = open(pickle_sfile, 'wb')
      save = {
        'train_dataset': sanitized_train_dataset,
        'train_labels': sanitized_train_label,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise



def evaluate_prediction(y_hat, y, target_names=None):

    #sklearn.metrics.accuracy_score(y, y_hat)
    print("Confusion matrix :")
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_hat)
    print(confusion_matrix)
    print("Classification report : ")
    print(sklearn.metrics.classification_report(y, y_hat, target_names=target_names))


import sklearn.linear_model
logisticRegressionClassifier = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
# train logistic
training_size = 500

logisticRegressionClassifier.fit(train_dataset[:training_size,:,:].reshape([training_size, 28*28]), train_labels[:training_size])
prediction = logisticRegressionClassifier.predict(valid_dataset.reshape([valid_size, 28*28]))
evaluate_prediction(prediction, valid_labels)

logisticRegressionClassifier.fit(sanitized_train_dataset[:training_size,:,:].reshape([training_size, 28*28]), sanitized_train_label[:training_size])
prediction = logisticRegressionClassifier.predict(valid_dataset.reshape([valid_size, 28*28]))
evaluate_prediction(prediction, valid_labels)