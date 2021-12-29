import tensorflow as tf
from tensorflow.keras.regularizers import l2

# HHAR number of classes
NUM_CLASSES = 6

# HHAR channels
NUM_CHANNELS = 6

# CNN model generator for HHAR using sequential
def generate_model_hhar():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(
        filters=16, kernel_size=5, activation='relu', input_shape=(256, NUM_CHANNELS),
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(
        filters=32, kernel_size=5, activation='relu',
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(
        filters=64, kernel_size=5, activation='relu',
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=256, activation='relu',
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(
        units=NUM_CLASSES,
    ))
    return model