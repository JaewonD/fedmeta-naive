import random
import numpy as np
from functools import reduce

from models.generate_model import generate_model
from utils.logger import Logger
from clients.clientdefault import ClientDefault

class ServerDefault():
    def __init__(self, train_ids, validation_ids, test_ids, dataset_name, note):
        self.train_ids = train_ids
        self.validation_ids = validation_ids
        self.test_ids = test_ids
        self.dataset_name = dataset_name
        self.eval_every = 1
        self.total_rounds = 1000
        self.num_fit_clients = 3
        self.model = generate_model(self.dataset_name)
        self.logger = Logger(f'{dataset_name}_default_{test_ids}_{note}')
        self.train_clients = [ClientDefault(self.dataset_name, i, self.logger) for i in self.train_ids]
        self.validation_clients = [ClientDefault(self.dataset_name, i, self.logger) for i in self.validation_ids]
        self.test_clients = [ClientDefault(self.dataset_name, i, self.logger) for i in self.test_ids]

    def run(self):
        for rnd in range(1, self.total_rounds + 1):
            print(f'Round: {rnd}')
            self.train(rnd)
            if rnd % self.eval_every == 0:
                self.test(rnd)

    def train(self, rnd):
        returned_weights = []
        selected_users = random.sample(self.train_clients, self.num_fit_clients)
        weight = self.model.get_weights()
        for c in selected_users:
            returned_weight, num_examples = c.train(rnd, weight)
            returned_weights.append((returned_weight, num_examples))
        aggregated_weight = self.aggregate(returned_weights)
        self.model.set_weights(aggregated_weight)

    def test(self, rnd):
        weight = self.model.get_weights()
        for c in self.train_clients:
            c.test(rnd, weight)
        for c in self.validation_clients:
            c.test(rnd, weight)
        for c in self.test_clients:
            c.test(rnd, weight)

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
