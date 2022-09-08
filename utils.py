import numpy as np
import torch
import pickle
import pandas as pd

def load_pkl(filename):
    with open(filename, "rb") as fi:
        res = pickle.load(fi)
    return res

def split_data(data):

    data = data.sort_values(['DATE', 'permno'])
    group_df = data.groupby('DATE')
    data_list = []
    for i in group_df:
        data_list.append(i[1].iloc[:, 2:].values)
    data = np.array(data_list)
    return data[:, :, :-2], data[:, :, -1]

def load_data():

    Subset = load_pkl("./Processed data/Six Datasets/Subset_2.pkl")
    industry_relation = load_pkl("./Processed data/Six Datasets relations/industry/Subset_2_industry_relation.pkl")
    location_relation = load_pkl("./Processed data/Six Datasets relations/location/Subset_2_location_relation.pkl")

    X = split_data(Subset)[0]
    y = split_data(Subset)[1]

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    industry_relation = torch.tensor(industry_relation, dtype=torch.float)
    location_relation = torch.tensor(location_relation, dtype=torch.float)

    return X, y, industry_relation, location_relation

def R2_score_calculate(true, pred):
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum(true ** 2)
    return r2

def IC_ICIR_score_calculate(true, pred, Length_time):
    df = pd.DataFrame()
    times = [time for time in list(range(1, Length_time + 1)) for i in range(int(len(true) / Length_time))]
    df['date'] = times
    df['true'] = true
    df['pred'] = pred
    rank_ic = df.groupby("date").apply(lambda x: x["true"].corr(x["pred"], method="spearman"))
    rank_ic = np.array(rank_ic)
    rank_ic_mean = np.mean(rank_ic)
    rank_ic_std = np.std(rank_ic, ddof=1)
    return rank_ic_mean, rank_ic_mean/rank_ic_std
