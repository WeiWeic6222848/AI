import os
import random

import numpy as np

from OLD import STANOLD, matrix_to_list
from Recommenders import *
import pandas
import pandas as pd
import datetime


base_models = [
    Popularity(),
    STAN(factor1=True, l1=6,
        factor2=True, l2=6000 * 365 * 24 * 3600, factor3=True, l3=6),
    STANOLD(factor1=True, l1=6,
            factor2=True, l2=6000 * 365 * 24 * 3600, factor3=True, l3=6)
]

# recipe = pandas.read_csv(
#    './RAW_recipes.csv')

def readCsv(fold):
    df = pandas.read_csv(
        f'./folds/{fold}/train.csv').drop(
        ['review'], axis=1)
    dftest = pandas.read_csv(
        f'./folds/{fold}/validate.csv').drop(
        ['review'], axis=1)

    # get the year from data
    df['rate_year'] = pandas.DatetimeIndex(df['date']).year
    dftest['rate_year'] = pandas.DatetimeIndex(dftest['date']).year

    # get the month from data
    df['rate_month'] = pandas.DatetimeIndex(df['date']).month
    dftest['rate_month'] = pandas.DatetimeIndex(dftest['date']).month

    # get season from month data
    # 1,2,3 -> season 1.
    # 4,5,6 -> season 2. etc.
    df['rate_season'] = (df['rate_month'] - 1) // 3 + 1
    dftest['rate_season'] = (dftest['rate_month'] - 1) // 3 + 1

    # session_id defined as user_id + rate_year and each session item contains recipes the user rated,
    # filter out session length with only 1 rated recipe
    # df = df.groupby(['user_id', 'rate_year']).filter(
    #     lambda x: len(x['recipe_id']) > 1)
    # dftest = dftest.groupby(['user_id', 'rate_year']).filter(
    #     lambda x: len(x['recipe_id']) > 1)

    # serialize session id to continous integer for matrix initialization
    df['session_id'] = pd.factorize(
        pd._libs.lib.fast_zip([df['user_id'].values, df['rate_year'].values]))[
        0]
    dftest['session_id'] = pd.factorize(
        pd._libs.lib.fast_zip(
            [dftest['user_id'].values, dftest['rate_year'].values]))[0]

    # we keep the map of recipe to retrieve the original recipe information to see if recommendation was good
    codes, recipe_map = pd.factorize(
        pd.concat([df['recipe_id'], dftest['recipe_id']]))
    df['recipe_id_serialized'] = codes[:len(df)]
    dftest['recipe_id_serialized'] = codes[len(df):]
    del codes

    # convert date of review to timestamp for test/train split as well as data purposes
    df['timestamp'] = df['date'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').timestamp())
    dftest['timestamp'] = dftest['date'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').timestamp())

    # sort by timestamp
    df = df.sort_values(by=['timestamp'])
    dftest = dftest.sort_values(by=['timestamp'])

    # calculate item sequence
    df['item_index'] = df.groupby(
        'session_id').cumcount() + 1
    dftest['item_index'] = dftest.groupby(
        'session_id').cumcount() + 1

    #drop all unnecessary informations
    df = df.drop(
        ['recipe_id', 'date', 'rating', 'count_user', 'count_item',
         'rate_year', 'rate_month', 'rate_season'], axis=1)
    dftest = dftest.drop(
        ['recipe_id', 'date', 'rating', 'count_user', 'count_item',
         'rate_year', 'rate_month', 'rate_season'], axis=1)

    return df, dftest, recipe_map


stats_dict = {}
for model in base_models:
    stats_dict[model.tostring()] = []

# read the dataset
for fold in os.listdir('folds'):

    # if fold == "fold_2":
    #    break

    print(f"Testing for {fold}")
    train_session, test_session, recipe_map = readCsv(fold)

    # convert to csr where row = session and column = item-to-sessions
    test_session_matrix = sparse.csr_matrix(
        (test_session['item_index'],
         (test_session['session_id'], test_session['recipe_id_serialized'])),
        shape=(test_session['session_id'].max() + 1,
               len(recipe_map)))
    # shape is max index + 1 to include zero index
    train_session_matrix = sparse.csr_matrix(
        (train_session['item_index'],
         (train_session['session_id'], train_session['recipe_id_serialized'])),
        shape=(train_session['session_id'].max() + 1,
               len(recipe_map)))

    # creating timestamp matrices for both train and
    idx = test_session.groupby('session_id')['timestamp'].transform('max') == \
          test_session['timestamp']
    test_timestamp = test_session[idx]
    test_timestamp = test_timestamp.drop_duplicates(subset=['session_id'])

    test_timestamp_matrix = sparse.csr_matrix(
        (test_timestamp['timestamp'],
         (test_timestamp['session_id'],
          [0 for _ in range(len(test_timestamp))])),
        # always col 0
        shape=(test_timestamp['session_id'].max() + 1,
               1))

    idx = train_session.groupby('session_id')['timestamp'].transform('max') == \
          train_session['timestamp']
    train_timestamp = train_session[idx]
    train_timestamp = train_timestamp.drop_duplicates(subset=['session_id'])

    train_timestamp_matrix = sparse.csr_matrix(
        (train_timestamp['timestamp'],
         (train_timestamp['session_id'],
          [0 for _ in range(len(train_timestamp))])),
        shape=(train_timestamp['session_id'].max() + 1,
               1))


    def split_seq(session_item_matrix, timestamp_matrix):
        preds = np.empty((0, 1), dtype=int)
        uniquerow = np.unique(session_item_matrix.nonzero()[0])
        for row in uniquerow:
            max_idx = session_item_matrix[row].argmax()
            preds = np.append(preds, max_idx)
            session_item_matrix[row, max_idx] = 0
        session_item_matrix.eliminate_zeros()
        return session_item_matrix, timestamp_matrix, preds


    test_session_matrix, test_timestamp_matrix, test_predict = split_seq(
        test_session_matrix, test_timestamp_matrix)

    # Calcultae metrics Recall, MRR and NDCG for each model
    for model in base_models:
        model.fit(train_session_matrix, train_timestamp_matrix)

        print("MODEL = ", model.tostring())
        testing_size = test_session_matrix.get_shape()[0]
        # testing_size = 10

        R_5 = 0
        R_10 = 0
        R_20 = 0

        NDCG_5 = 0
        NDCG_10 = 0
        NDCG_20 = 0

        predictions = model.predict(test_session_matrix,
                                    test_timestamp_matrix)
        for i in range(testing_size):

            # for s in score:
            #     print(s)
            # print(test_predict[i])
            # print("-----------------------------------")
            # print("-----------------------------------")
            items = predictions[i][0]
            # if len(items) == 0:
            #     print("!!!")
            if test_predict[i] in items:
                rank = int(
                    np.where(items == test_predict[i])[0]) + 1
                # print(rank)
                R_20 += 1
                NDCG_20 += 1 / math.log(rank + 1, 2)

                if rank <= 5:
                    R_5 += 1
                    NDCG_5 += 1 / math.log(rank + 1, 2)

                if rank <= 10:
                    R_10 += 1
                    NDCG_10 += 1 / math.log(rank + 1, 2)

        R_5 = R_5 / testing_size
        R_10 = R_10 / testing_size
        R_20 = R_20 / testing_size
        NDCG_5 = NDCG_5 / testing_size
        NDCG_10 = NDCG_10 / testing_size
        NDCG_20 = NDCG_20 / testing_size

        print("R@5: %f" % R_5)
        print("R@10: %f" % R_10)
        print("R@20: %f" % R_20)
        print("NDCG@5: %f" % NDCG_5)
        print("NDCG@10: %f" % NDCG_10)
        print("NDCG@20: %f" % NDCG_20)
        print(
            "training size: %d" % train_session_matrix.get_shape()[
                0])
        print("testing size: %d" % test_session_matrix.get_shape()[
            0])

        stats = {
            "R@5": R_5,
            "R@10": R_10,
            "R@20": R_20,
            "NDCG@5": NDCG_5,
            "NDCG@10": NDCG_10,
            "NDCG@20": NDCG_20,
        }

        stats_dict[model.tostring()].append(stats)

print(stats_dict)
finalstats = []
for model in base_models:
    print(f"final statistic for model {model.tostring()}")
    for key in stats_dict[model.tostring()][0].keys():
        statsList = list(map(lambda x: x[key], stats_dict[model.tostring()]))
        mean = np.mean(statsList)
        std = np.std(statsList)
        print(f"{key} stats: mean = {mean}, std = {std}")
