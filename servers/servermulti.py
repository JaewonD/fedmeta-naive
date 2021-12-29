import random
import numpy as np
import tensorflow as tf
from functools import reduce

from models.generate_model import generate_model
from utils.logger import Logger
from clients.clientmulti import ClientMulti

class ServerMulti():
    def __init__(self, test_id, dataset_name, note):
        self.total_num_clients = 10
        self.test_id = test_id
        self.validation_id = (test_id + 9) % self.total_num_clients
        self.train_ids = list(range(self.total_num_clients))
        self.train_ids.remove(test_id)
        self.train_ids.remove(self.validation_id)
        self.dataset_name = dataset_name
        self.eval_every = 1
        self.total_rounds = 500
        self.num_fit_clients = 3
        self.num_fit_multi_tasks = 3
        initial_model = generate_model(self.dataset_name)
        self.weight = initial_model.get_weights()
        tf.keras.backend.clear_session()
        self.logger = Logger(f'{dataset_name}_multi_{self.test_id}_{note}')
        self.clients = [ClientMulti(self.dataset_name, i, self.logger) for i in range(self.total_num_clients)]
        self.num_class_partitions = 3
        if self.dataset_name == 'ICHAR':
            self.num_classes = 9
        elif self.dataset_name == 'ICSR':
            self.num_classes = 14

    def run(self):
        for rnd in range(1, self.total_rounds + 1):
            print(f'Round: {rnd}')
            self.train(rnd)
            if rnd % self.eval_every == 0:
                self.test(rnd)

    def train(self, rnd):
        returned_weights = []
        # Per-condition tasks training
        selected_user_id = random.sample(self.train_ids, self.num_fit_clients)
        for train_id in selected_user_id:
            c = self.clients[train_id]
            returned_weight = c.per_train(rnd, self.weight)
            returned_weights.append((returned_weight, 1))
        # Multi-conditioned tasks training
        for _ in range(self.num_fit_multi_tasks):
            selected_user_id = random.sample(self.train_ids, self.num_class_partitions)
            class_partitions = self.get_class_partition(self.num_classes, self.num_class_partitions)
            # Meta-train
            meta_train_returned_weights = []
            for (i, train_id) in enumerate(selected_user_id):
                c = self.clients[train_id]
                returned_weight = c.multi_train(rnd, self.weight, class_partitions[i])
                meta_train_returned_weights.append((returned_weight, 1))
            meta_train_weight_aggr = self.aggregate(meta_train_returned_weights)
            # Meta-update
            meta_update_returned_weights = []
            for (i, train_id) in enumerate(selected_user_id):
                c = self.clients[train_id]
                returned_weight = c.multi_update(rnd, meta_train_weight_aggr, class_partitions[i])
                meta_update_returned_weights.append((returned_weight, 1))
            meta_update_weight_aggr = self.aggregate(meta_update_returned_weights)
            final_weight = self._add_weights(self.weight, self._subtract_weights(meta_update_weight_aggr, meta_train_weight_aggr))
            returned_weights.append((final_weight, 1))
        self.weight = self.aggregate(returned_weights)

    def test(self, rnd):
        for test_id in range(self.total_num_clients):
            c = self.clients[test_id]
            c.test(rnd, self.weight)

    def aggregate(self, results):
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])
        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]
        # Compute average weights of each layer
        weights_prime = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    def get_class_partition(self, num_classes, num_partitions):  
        lst = list(range(num_classes))
        random.shuffle(lst)
        division = len(lst) / num_partitions
        return [lst[round(division * i):round(division * (i + 1))] for i in range(num_partitions)]
    
    def _add_weights(self, weight1, weight2):
        return [p1 + p2 for p1, p2 in zip(weight1, weight2)]

    def _subtract_weights(self, weight1, weight2):
        return [p1 - p2 for p1, p2 in zip(weight1, weight2)]