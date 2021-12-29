import pandas as pd
import json
from collections import Counter

WIN_LEN = 8
users = ['S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
classes = ['amusement', 'baseline', 'mediation', 'stress']

if __name__ == "__main__":
    df = pd.read_csv('data_WESAD/both_minmax_scaled_all.csv')

    for (userid, user) in enumerate(users):
        dataset = {'x': [], 'y': []}
        for (classid, label) in enumerate(classes):
            filtered_data = df[(df["domain"] == user) & (df["label"] == label)]
            i = 0
            while i + WIN_LEN - 1 < len(filtered_data):
                feature = filtered_data.iloc[i:i+WIN_LEN, 0:10].values
                feature = feature.tolist()
                dataset['x'].append(feature)
                dataset['y'].append(classid)
                i += WIN_LEN
        file_path = f'data_WESAD/id/{userid}.json'
        with open(file_path, 'w') as outfile:
            json.dump(dataset, outfile)
 