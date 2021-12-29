import argparse
import os
import tensorflow as tf
import random
import numpy as np

def get_configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument('--note',
                    help='label to add a note to experiment;',
                    type=str,
                    required=True)
    parser.add_argument('--algorithm',
                    help='algorithm to run experiment;',
                    type=str,
                    required=True)
    parser.add_argument('--testid',
                    help='test user ID for ICHAR;',
                    type=str,
                    required=True)
    parser.add_argument('--dataset',
                    help='dataset name;',
                    type=str,
                    required=True)   
    args = parser.parse_args()
    return args

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if __name__ == "__main__":
    args = get_configuration()

    # train/val/test definition
    if args.dataset in ['ICHAR', 'ICSR']:
        test_id = int(args.testid)
        validation_id = (test_id + 9) % 10
        train_ids = list(range(10))
        train_ids.remove(test_id)
        train_ids.remove(validation_id)
        # Convert to str
        train_ids = list(map(str, train_ids))
        validation_ids = [str(validation_id)]
        test_ids = [str(test_id)]
    elif args.dataset in ['WESAD']:
        test_id = int(args.testid)
        validation_id = (test_id + 14) % 15
        train_ids = list(range(15))
        train_ids.remove(test_id)
        train_ids.remove(validation_id)
        # Convert to str
        train_ids = list(map(str, train_ids))
        validation_ids = [str(validation_id)]
        test_ids = [str(test_id)]
    elif args.dataset in ['HHAR']:
        test_id_u, test_id_d = int(args.testid[1]), int(args.testid[4])
        validation_id_u = (test_id_u + 5) % 6
        validation_id_d = (test_id_d + 4) % 5
        train_ids_u = list(range(6))
        train_ids_u.remove(test_id_u)
        train_ids_u.remove(validation_id_u)
        train_ids_d = list(range(5))
        train_ids_d.remove(test_id_d)
        train_ids_d.remove(validation_id_d)
        # Convert to str
        train_ids = [f'u{u}_d{d}' for u in train_ids_u for d in train_ids_d]
        validation_ids = [f'u{validation_id_u}_d{validation_id_d}']
        test_ids = [f'u{test_id_u}_d{test_id_d}']

    # server definition
    if args.algorithm == 'default':
        from servers.serverdefault import ServerDefault
        server = ServerDefault(train_ids, validation_ids, test_ids, args.dataset, args.note)
    elif args.algorithm == 'meta':
        from servers.servermeta import ServerMeta
        server = ServerMeta(train_ids, validation_ids, test_ids, args.dataset, args.note)
    elif args.algorithm == 'mix':
        from servers.servermix import ServerMix
        server = ServerMix(train_ids, validation_ids, test_ids, args.dataset, args.note)
    
    server.run()
