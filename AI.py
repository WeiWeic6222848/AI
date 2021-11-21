import random

import numpy as np

from Recommenders import *
import pandas
import pandas as pd
import datetime

# read the dataset
df = pandas.read_csv(
    './RAW_interactions.csv').drop(
    ['review'], axis=1)[:10000]
# recipe = pandas.read_csv(
#    './RAW_recipes.csv')

# get the year from year data
df['rate_year'] = pandas.DatetimeIndex(df['date']).year

# session_id defined as user_id + rate_year and each session item contains recipes the user rated, filter out session length with only 1 rated recipe
df = df.groupby(['user_id', 'rate_year']).filter(
    lambda x: len(x['recipe_id']) > 1)

# serialize session id to continous integer for matrix initialization
df['session_id'] = pd.factorize(
    pd._libs.lib.fast_zip([df['user_id'].values, df['rate_year'].values]))[0]

# test train split
# first, randomly take 10% of the sessions out from the dataset as a way to vary the data to create a different testing situation each run
sessionRange = df['session_id'].max()
numToRemove = random.sample(range(0, sessionRange), sessionRange // 10)

# remove the session with given id from the dataset
df = df.loc[~df['session_id'].isin(numToRemove)]





# re-serialize the sesion id
df = df.reset_index(drop=True)
# df['session_id'] = pd.factorize(
#     pd._libs.lib.fast_zip([df['user_id'].values, df['rate_year'].values]))[0]

# serialize the recipe id such that the resulting sparse matrix is smaller

# we keep the map of recipe to retrieve the original recipe information to see if recommendation was good
df['recipe_id_serialized'], recipe_map = pd.factorize(df['recipe_id'].values)

# convert date of review to timestamp for test/train split as well as data purposes
df['timestamp'] = df['date'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').timestamp())

# test session is equal to 20% of the current dataset

# take 20% session with the largest timestamp
test_session_id = df.groupby('session_id', as_index=False).max()
test_session_id = test_session_id.nlargest(len(test_session_id) // 20,
                                           'timestamp')['session_id']



# definitive split test and train
test_session = df.loc[df['session_id'].isin(test_session_id)].reset_index(
    drop=True)
train_session = df.loc[~df['session_id'].isin(test_session_id)].reset_index(
    drop=True)

# reset session id
test_session['session_id'] = pd.factorize(test_session['session_id'].values)[0]
train_session['session_id'] = pd.factorize(train_session['session_id'].values)[0]

# order by timestamp
test_session = test_session.sort_values(by=['timestamp'])
train_session = train_session.sort_values(by=['timestamp'])

# convert to csr where row = session and column = item-to-sessions
test_session_matrix = sparse.coo_matrix(
    ([1 for _ in range(len(test_session['rating']))],
     (test_session['session_id'], test_session['recipe_id_serialized'])),
    shape=(test_session['session_id'].max() + 1,
           df['recipe_id_serialized'].max() + 1))
# shape is max index + 1 to include zero index

train_session_matrix = sparse.coo_matrix(
    ([1 for _ in range(len(train_session['rating']))],
     (train_session['session_id'], train_session['recipe_id_serialized'])),
    shape=(train_session['session_id'].max() + 1,
           df['recipe_id_serialized'].max() + 1))

# creating timestamp matrices for both train and
idx = test_session.groupby('session_id')['timestamp'].transform('max') == \
      test_session['timestamp']
test_timestamp = test_session[idx]

test_timestamp_matrix = sparse.csr_matrix(
    (test_timestamp['timestamp'],
     (test_timestamp['session_id'], [0 for _ in range(len(test_timestamp))])),
    # always col 0
    shape=(test_timestamp['session_id'].max() + 1,
           1))

idx = train_session.groupby('session_id')['timestamp'].transform('max') == \
      train_session['timestamp']
train_timestamp = train_session[idx]

train_timestamp_matrix = sparse.csr_matrix(
    (train_timestamp['timestamp'],
     (
         train_timestamp['session_id'],
         [0 for _ in range(len(train_timestamp))])),
    shape=(train_timestamp['session_id'].max() + 1,
           1))


# import pickle
# import numpy as np
# import math
# import time
#
# model = SKNN(session_id=train_id, session=train_session, session_timestamp=train_timestamp, sample_size=0, k=500)
# model = VSKNN(session_id=train_id, session=train_session, session_timestamp=train_timestamp, sample_size=0, k=500)[
# STAN and Popularity recommender models

def split_seq(session_item_matrix, timestamp_matrix):
    new_coomatrix_input = np.empty((0, 3), dtype=int)
    preds = np.empty((0, 1), dtype=int)
    time_matrix_input = np.empty((0, 3), dtype=int)
    session_ids = 0
    for row in session_item_matrix.nonzero()[0]:
        sequence_information = session_item_matrix.col[
            session_item_matrix.row == row]
        while len(sequence_information) > 1:
            for col in sequence_information[:-1]:
                new_coomatrix_input = np.vstack([new_coomatrix_input, np.array(
                    [session_item_matrix.getrow(row).getcol(col).data[0],
                     session_ids, col])])
            preds = np.vstack([preds, np.array([sequence_information[-1]])])

            time_matrix_input = np.vstack([time_matrix_input, np.array([
                timestamp_matrix.getrow(row).data[0],
                session_ids, sequence_information[-1]])])

            session_ids += 1
            sequence_information = sequence_information[:-1]

    new_coomatrix = sparse.coo_matrix((new_coomatrix_input[:, 0], (
        new_coomatrix_input[:, 1], new_coomatrix_input[:, 2])), shape=(
        int(new_coomatrix_input[:, 1].max() + 1),
        session_item_matrix.get_shape()[1]))

    new_timematrix = sparse.coo_matrix((time_matrix_input[:, 0], (
        time_matrix_input[:, 1], [0 for _ in range(len(time_matrix_input))])),
                                       shape=(
                                           int(new_coomatrix_input[:,
                                               1].max() + 1), 1))
    return new_coomatrix, new_timematrix, preds


test_session_matrix, test_timestamp_matrix, test_predict = split_seq(
    test_session_matrix, test_timestamp_matrix)

models = [
    # Popularity().fit(train_session_matrix),
    STAN().fit(train_session_matrix, train_timestamp_matrix)
]

# Calcultae metrics Recall, MRR and NDCG for each model
for model in models:
    print("MODEL = ", model)
    testing_size = len(test_session)
    # testing_size = 10

    R_5 = 0
    R_10 = 0
    R_20 = 0

    MRR_5 = 0
    MRR_10 = 0
    MRR_20 = 0

    NDCG_5 = 0
    NDCG_10 = 0
    NDCG_20 = 0
    for i in range(testing_size):
        if i % 1000 == 0:
            print("%d/%d" % (i, testing_size))
            # print("MRR@20: %f" % (MRR_20 / (i + 1)))

        score = model.predict(
            test_session_matrix.getrow(i), test_timestamp_matrix.getrow(i),
            test_session_matrix.col[test_session_matrix.row == i]
        )
        # for s in score:
        #     print(s)
        # print(test_predict[i])
        # print("-----------------------------------")
        # print("-----------------------------------")
        items = [x[0] for x in score]
        # if len(items) == 0:
        #     print("!!!")
        if test_predict[i] in items:
            rank = items.index(test_predict[i]) + 1
            # print(rank)
            MRR_20 += 1 / rank
            R_20 += 1
            NDCG_20 += 1 / math.log(rank + 1, 2)

            if rank <= 5:
                MRR_5 += 1 / rank
                R_5 += 1
                NDCG_5 += 1 / math.log(rank + 1, 2)

            if rank <= 10:
                MRR_10 += 1 / rank
                R_10 += 1
                NDCG_10 += 1 / math.log(rank + 1, 2)

                # print("past recipes:")
                # for r in test_session[i]:
                #     print(recipe[recipe['id'] == r]['name'].tolist()[0],
                #           end=",\n")
                # print("\n")
                # print("recommended recipes:")
                # for r in items:
                #     print(recipe[recipe['id'] == r]['name'].tolist()[0],
                #           end=",\n")
                # print("\n")
                # print("actual next recipe:")
                # print(recipe[recipe['id'] == test_predict[i]]['name'].tolist()[
                #           0])
                # print("\n\n\n\n\n")

    MRR_5 = MRR_5 / testing_size
    MRR_10 = MRR_10 / testing_size
    MRR_20 = MRR_20 / testing_size
    R_5 = R_5 / testing_size
    R_10 = R_10 / testing_size
    R_20 = R_20 / testing_size
    NDCG_5 = NDCG_5 / testing_size
    NDCG_10 = NDCG_10 / testing_size
    NDCG_20 = NDCG_20 / testing_size

    print("MRR@5: %f" % MRR_5)
    print("MRR@10: %f" % MRR_10)
    print("MRR@20: %f" % MRR_20)
    print("R@5: %f" % R_5)
    print("R@10: %f" % R_10)
    print("R@20: %f" % R_20)
    print("NDCG@5: %f" % NDCG_5)
    print("NDCG@10: %f" % NDCG_10)
    print("NDCG@20: %f" % NDCG_20)
    print("training size: %d" % len(train_session))
    print("testing size: %d" % testing_size)
