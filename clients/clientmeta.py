import numpy as np
import tensorflow as tf
import json

from models.generate_model import generate_model
from utils.lr import lr

class ClientMeta():
    def __init__(self, dataset_name, cid, logger):
        self.dataset_name = dataset_name
        self.cid = cid
        self.logger = logger
        file_path = f'./data_{dataset_name}/id/{cid}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.data_x = np.array(data['x'])
            self.data_y = np.array(data['y'])
        self.inner_lr = lr[self.dataset_name]['inner_lr']
        self.outer_lr = lr[self.dataset_name]['outer_lr']
        self.num_shots = 3
        self.inner_step = 5
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)
        self.inner_optimizer = tf.keras.optimizers.SGD(learning_rate=self.inner_lr)
        self.outer_repetition = 10
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    def train(self, rnd, weights):
        tf.keras.backend.clear_session()
        model = generate_model(self.dataset_name)
        model.set_weights(weights)
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=self.outer_lr)
        for _ in range(self.outer_repetition):
            support_x, support_y, query_x, query_y = self._sample_two_chunks_k1_k2_shots(self.data_x, self.data_y, self.num_shots, self.num_shots)
            with tf.GradientTape() as t_outer:
                # Take one gradient step
                with tf.GradientTape() as t_inner:
                    logits = model(support_x, training=True)
                    loss = self.loss_fn(support_y, logits)
                grad = t_inner.gradient(loss, model.trainable_variables)
                # Copy model and calculate adapted model
                model_copy = generate_model(self.dataset_name)
                k = 0
                for i in range(len(model_copy.layers)):
                    if model_copy.layers[i].name.startswith('conv') or model_copy.layers[i].name.startswith('dense'):
                        model_copy.layers[i].kernel = tf.subtract(model.layers[i].kernel, tf.multiply(self.inner_lr, grad[k]))
                        model_copy.layers[i].bias = tf.subtract(model.layers[i].bias, tf.multiply(self.inner_lr, grad[k + 1]))
                        k += 2
                    if model_copy.layers[i].name.startswith('batch'):
                        model_copy.layers[i].gamma = tf.subtract(model.layers[i].gamma, tf.multiply(self.inner_lr, grad[k]))
                        model_copy.layers[i].beta = tf.subtract(model.layers[i].beta, tf.multiply(self.inner_lr, grad[k + 1]))
                        k += 2
                # Calculate meta loss
                logits = model_copy(query_x, training=True)
                loss = self.loss_fn(query_y, logits)
            grad = t_outer.gradient(loss, model.trainable_variables)
            self.outer_optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return model.get_weights()

    def test(self, rnd, weights):
        tf.keras.backend.clear_session()
        model = generate_model(self.dataset_name)
        model.set_weights(weights)
        adaptation_x, adaptation_y, remaining_x, remaining_y = self._split_into_k_shots_and_remainder(self.data_x, self.data_y, self.num_shots)
        # Adapt the model
        for _ in range(self.inner_step):
            with tf.GradientTape() as t_inner:
                logits = model(adaptation_x, training=True)
                loss = self.loss_fn(adaptation_y, logits)
            grad = t_inner.gradient(loss, model.trainable_variables)
            self.inner_optimizer.apply_gradients(zip(grad, model.trainable_variables))
        # Evaluate the adapted model
        logits = model(remaining_x, training=True) # training=True to not use moving statistics in batchnorm
        loss_value = self.loss_fn(remaining_y, logits)
        loss_value = float(tf.reduce_mean(loss_value).numpy())
        self.metric_accuracy.update_state(remaining_y, logits)
        accuracy = float(self.metric_accuracy.result().numpy())
        self.metric_accuracy.reset_states()
        self.logger.log_test_data(round_number=rnd, cid=self.cid, num_samples=len(remaining_x), loss=loss_value, accuracy=accuracy)
        tf.keras.backend.clear_session()

    def _split_into_k_shots_and_remainder(self, data_x, data_y, k):
        # Return one chunk with k shots and another chunk with remaining data.
        available_classes = list(set(data_y.tolist()))
        shot_indices = []
        for i in available_classes:
            indices_in_class = []
            for j in range(len(data_y)):
                if data_y[j] == i:
                    indices_in_class.append(j)
            k_shot_indices_in_class = np.random.choice(indices_in_class, size=k, replace=False)
            shot_indices.extend(k_shot_indices_in_class.tolist())
        shot_indices = np.array(shot_indices)
        remaining_indices = np.setdiff1d(np.arange(len(data_y)), shot_indices)
        return data_x[shot_indices], data_y[shot_indices], data_x[remaining_indices], data_y[remaining_indices]

    def _sample_two_chunks_k1_k2_shots(self, data_x, data_y, k1, k2):
        chunk1_x, chunk1_y, remaining_x, remaining_y = self._split_into_k_shots_and_remainder(data_x, data_y, k1)
        chunk2_x, chunk2_y, _, _ = self._split_into_k_shots_and_remainder(remaining_x, remaining_y, k2)
        return chunk1_x, chunk1_y, chunk2_x, chunk2_y