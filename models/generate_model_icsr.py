import tensorflow as tf
from tensorflow.keras.regularizers import l2

# ICSR number of classes
NUM_CLASSES = 14

# ICSR channels
NUM_CHANNELS = 1

# CNN model generator for ICSR using sequential
def generate_model_icsr():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(
        filters=16, kernel_size=6, strides=3, activation='relu', input_shape=(32000, NUM_CHANNELS),
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(
        filters=32, kernel_size=6, strides=3, activation='relu',
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(
        filters=64, kernel_size=6, strides=2, activation='relu',
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