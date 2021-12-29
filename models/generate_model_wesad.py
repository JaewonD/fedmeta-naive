import tensorflow as tf
from tensorflow.keras.regularizers import l2

# WESAD number of classes
NUM_CLASSES = 4

# WESAD channels
NUM_CHANNELS = 10

# CNN model generator for WESAD using sequential
def generate_model_wesad():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(
        filters=32, kernel_size=2, activation='relu', input_shape=(8, NUM_CHANNELS),
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(
        filters=64, kernel_size=2, activation='relu',
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(
        filters=128, kernel_size=2, activation='relu',
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=NUM_CLASSES,
    ))
    return model