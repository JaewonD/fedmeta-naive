from typing import Dict
import os
import json

from utils.log_keys import (
    NUM_ROUND_KEY,
    ACCURACY_KEY,
    LOSS_KEY,
    CLIENT_ID_KEY,
    NUM_SAMPLES_KEY,
)

class Logger():
    # Initializes logger.
    def __init__(self, name):
        # Initialize file names
        self.test_file_name = 'test.csv'
        # Initialize log folder path
        self.log_folder = os.path.join('.', 'log', name)
        os.makedirs(self.log_folder)
        # Initialize log keys order. Logs are written in this order.
        self.log_keys_list = [NUM_ROUND_KEY, CLIENT_ID_KEY, NUM_SAMPLES_KEY, LOSS_KEY, ACCURACY_KEY]
        # Create initial files
        self._write_test_file_header()

    def _join_strings_by_comma(self, strings):
        return ','.join(strings) + '\n'

    def _write_test_file_header(self):
        with open(os.path.join(self.log_folder, self.test_file_name), 'a') as test_file:
            test_file.write(self._join_strings_by_comma(self.log_keys_list))

    def _get_log_string(self, log_data_dict: Dict[str, str]):
        log_data_list = list(map(lambda key: log_data_dict[key], self.log_keys_list))
        return self._join_strings_by_comma(log_data_list)

    def log_test_data(self, round_number, cid, num_samples, loss, accuracy):
        if self.log_folder is None:
            return
        with open(os.path.join(self.log_folder, self.test_file_name), 'a') as test_file:
            log_data_dict = {
                NUM_ROUND_KEY: str(round_number),
                CLIENT_ID_KEY: str(cid),
                NUM_SAMPLES_KEY: str(num_samples),
                LOSS_KEY: str(loss),
                ACCURACY_KEY: str(accuracy),}
            test_file.write(self._get_log_string(log_data_dict))
