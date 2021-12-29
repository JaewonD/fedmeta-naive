import pandas as pd
import json
from collections import Counter

WIN_LEN = 256
users = ['a', 'b', 'c', 'd', 'e', 'g']
devices = ['gear', 'lgwatch', 'nexus4', 's3', 's3mini']
classes = ['bike', 'sit', 'stairsdown', 'stairsup', 'stand', 'walk']

if __name__ == "__main__":
    df = pd.read_csv('data_HHAR/hhar_std_scaling_all.csv')

    for (userid, user) in enumerate(users):
        for (deviceid, device) in enumerate(devices):
            dataset = {'x': [], 'y': []}
            for (classid, label) in enumerate(classes):
                filtered_data = df[(df["User"] == user) & (df["Model"] == device) & (df["gt"] == label)]
                i = 0
                while i + WIN_LEN - 1 < len(filtered_data):
                    feature = filtered_data.iloc[i:i+WIN_LEN, 3:9].values
                    feature = feature.tolist()
                    dataset['x'].append(feature)
                    dataset['y'].append(classid)
                    i += WIN_LEN // 2
            file_path = f'data_HHAR/id/u{userid}_d{deviceid}.json'
            with open(file_path, 'w') as outfile:
                json.dump(dataset, outfile)
 