import numpy as np
import tensorflow as tf
import json

from models.generate_model import generate_model
from utils.lr import lr

class ClientDefault():
    def __init__(self, dataset_name, cid, logger):
        self.dataset_name = dataset_name
        self.cid = cid
        self.logger = logger
        file_path = f'./data_{dataset_name}/id/{cid}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.data_x = np.array(data['x'])
            self.data_y = np.array(data['y'])
        self.model = generate_model(self.dataset_name)
        self.lr = lr[self.dataset_name]['lr']
        self.batch_size = 64
        self.epochs = 10
        self.dataset = tf.data.Dataset.from_tensor_slices((self.data_x, self.data_y))
        self.dataset = self.dataset.shuffle(buffer_size=4096).batch(self.batch_size)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    def train(self, rnd, weights):
        self.model.set_weights(weights)
        for _ in range(self.epochs):
            for (x_batch_train, y_batch_train) in self.dataset:
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return self.model.get_weights(), len(self.data_x)

    def test(self, rnd, weights):
        self.model.set_weights(weights)
        logits = self.model(self.data_x, training=True) # training=True to not use moving statistics in batchnorm
        loss_value = self.loss_fn(self.data_y, logits)
        loss_value = float(tf.reduce_mean(loss_value).numpy())
        self.metric_accuracy.update_state(self.data_y, logits)
        accuracy = float(self.metric_accuracy.result().numpy())
        self.metric_accuracy.reset_states()
        self.logger.log_test_data(round_number=rnd, cid=self.cid, num_samples=len(self.data_x), loss=loss_value, accuracy=accuracy)
