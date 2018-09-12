# List images and their labels
from random import shuffle
import glob
import cv2
import sys
import numpy as np
import tensorflow as tf

shuffle_data = True  # shuffle the addresses before saving
training_data_path ='dataset/*/*.jpg'

# read addresses and labels from the 'train' folder
addrs = glob.glob(training_data_path)
labels = [0 if 'dagim_kosher' in addr else 1 for addr in addrs]  # dagim_kosher: 0; taanug_kosher

# to shuffle data
if shuffle_data:
	c = list(zip(addrs, labels))
	shuffle(c)
	addrs, labels = zip(*c)

# divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

# Create a TFRecords file
# A function to Load images
def load_image(addr):
	# read an image and resize to (224, 224)
	# cv2 load images as BGR, convert it to RGB
	img = cv2.imread(addr)
	if img is None:
        return None
	img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.astype(np.float32)
	return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Write data into a TFRecord file
train_filename = 'train.tfrecords'

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_addrs)):
	# print how many images are saved every 1000 images
	if not i % 1000:
		print('Train data: {}/{}'.format(i, len(train_addrs)))
		sys.stdout.flush()

	# load the image
	img = load_image(train_addrs[i])

	label = train_labels[i]

	# create a feature
	feature = {'train/label': _int64_feature(label),
			   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
	
	# Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()

# open the TFRecords file
val_filename = 'val.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(val_filename)
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'Val data: {}/{}'.format(i, len(val_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(val_addrs[i])
    label = val_labels[i]
    # Create a feature
    feature = {'val/label': _int64_feature(label),
               'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()
# open the TFRecords file
test_filename = 'test.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'Test data: {}/{}'.format(i, len(test_addrs))
        sys.stdout.flush()
    # Load the image
    img = load_image(test_addrs[i])
    label = test_labels[i]
    # Create a feature
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()

