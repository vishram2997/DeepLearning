
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets    as tfds

tfds.disable_progress_bar()

(traing_data, test_data) = tfds.load('imdb_reviews/subwords8k', 
                                     split = (tfds.Split.TRAIN, tfds.Split.TEST),
                                     with_info = True, as_supervised = True)

encoder = info.features['text'].encoder
encoder.subwords[:20]