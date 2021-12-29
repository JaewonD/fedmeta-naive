import pandas as pd
import json
from collections import Counter

WIN_LEN = 32000
users = ['PH0007-jskim', 'PH0012-thanh', 'PH0014-wjlee', 'PH0034-ykha', 'PH0038-iygoo', 'PH0041-hmkim', 'PH0045-sjlee', 'WA0002-bkkim', 'WA0003-hskim', 'WA4697-jhryu']
classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off',
           'stop', 'go', 'forward',  'backward', 'follow', 'learn']

if __name__ == "__main__":
    df = pd.read_csv('data_ICSR/icsr_minmax_scaling_all.csv')

    dataset = [{'x': [], 'y': []} for i in range(len(users))]
    for idx in range(max(len(df) // WIN_LEN, 0)):
        cid = users.index(df.iloc[idx * WIN_LEN, 2])
        classid = classes.index(df.iloc[idx * WIN_LEN, 1])
        feature = df.iloc[idx * WIN_LEN:(idx + 1) * WIN_LEN, 0:1].values
        feature = feature.tolist()

        dataset[cid]['x'].append(feature)
        dataset[cid]['y'].append(classid)
    
    dest_folder = 'data_ICSR'
    for cid in range(len(users)):
        file_path = f'./{dest_folder}/id/{cid}.json'
        with open(file_path, 'w') as outfile:
            json.dump(dataset[cid], outfile)
    