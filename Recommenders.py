import math
from collections import Counter

import numpy
import numpy as np
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse, spatial


def max_n(row_data, row_indices, n):
    i = row_data.argsort()[-n:]
    # i = row_data.argpartition(-n)[-n:]
    top_values = row_data[i]
    top_indices = row_indices[i]  # do the sparse indices matter?
    return top_values, top_indices


# popularity recommender
class Popularity():
    def __init__(self, K=20):
        self.K = K

    def fit(self, X):
        # X = X.multiply(X >= 3)  # look only at items with rating more than 3
        items = list(X.nonzero()[1])
        sorted_scores = Counter(items).most_common()
        self.sorted_scores_ = [
            (item, score / sorted_scores[0][1]) for item, score in
            sorted_scores
        ]
        return self

    def predict(self, X):
        items, values = zip(*self.sorted_scores_[: self.K])

        users = set(X.nonzero()[0])

        U, I, V = [], [], []

        for user in users:
            U.extend([user] * self.K)
            I.extend(items)
            V.extend(values)

        score_matrix = scipy.sparse.csr_matrix((V, (U, I)), shape=X.shape)
        return score_matrix


class STAN:
    def __init__(self, k=20, sample_size=0,
                 kneighbor=500, factor1=True, l1=3.54,
                 factor2=True, l2=20 * 24 * 3600, factor3=True, l3=3.54, ):
        self.k = k
        self.kneighbor = kneighbor
        self.sample_size = sample_size
        self.factor1 = factor1
        self.factor2 = factor2
        self.factor3 = factor3
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.current_session_weight_cache = {}
        self.current_timestamp = 0

    def fit(self, session, session_timestamp):
        self.sequence_info = session
        self.session = session.copy()
        self.session.data[self.session.data != 1] = 1
        self.session_timestamp = session_timestamp
        return self

    def find_neighbours(self):
        similarity = sklearn.metrics.pairwise.cosine_similarity(
            self.current_session_weight_cache,
            self.session)
        similarity = scipy.sparse.csr_matrix(similarity)

        similarityNorm = similarity.copy()
        similarityNorm[similarityNorm.nonzero()] = 1

        if self.factor2 is True:
            timestampdata = scipy.sparse.hstack(
                [self.session_timestamp for _ in range(
                    self.current_timestamp.get_shape()[
                        0])]).transpose().tocsr().multiply(
                similarityNorm).tocsr()
            timestampdata.eliminate_zeros()
            nz = timestampdata.nonzero()
            timestampdata[nz] -= self.current_timestamp[nz[0]].transpose()

            # difference with original algorithm:
            # instead of plain subtraction, also take a abs value then negate all
            # to avoid positive numbers which ruins the weight process.

            timestampdata.data = numpy.exp( - numpy.abs(timestampdata.data) / self.l2)
            similarity = similarity.multiply(timestampdata)

        rows = np.unique(similarity.nonzero()[0])
        for rowindex in rows:
            row = similarity[rowindex]
            nz = row.nonzero()
            kneighbor, kneighborindex = max_n(similarity[rowindex, nz[1]].data,
                                              nz[1], self.kneighbor)
            not_top_k = [i for i in nz[1] if i not in kneighborindex]
            # remove neighbors not in top-k
            similarity[rowindex, not_top_k] = 0

        similarity.eliminate_zeros()
        return similarity

    # scoring item according to their session
    def score_items(self, neighbours):
        neighbours_normalized = neighbours.copy()
        neighbours_normalized[neighbours_normalized.nonzero()] = 1

        for session_idx, neighbour_session_map in enumerate(
                neighbours_normalized):

            if self.factor3 is True:
                neighbour_session = self.sequence_info.multiply(
                    neighbour_session_map.transpose())

                common_items = self.current_session[session_idx].multiply(
                    neighbour_session)

                # set items score of item present in the current session
                nz = common_items.nonzero()
                neighbour_session[nz] = 0


                common_items = common_items.max(axis=1).tocsr()

                nz = neighbour_session.nonzero()
                neighbour_session[nz] -= common_items[nz[0]].transpose()
                neighbour_session.eliminate_zeros()
                neighbour_session.data = numpy.exp(
                    - np.abs(neighbour_session.data) / self.l3)
                neighbour_session = neighbour_session.multiply(
                    neighbours[session_idx].transpose())
            else:
                neighbour_session = self.session.multiply(
                    neighbours[session_idx].transpose())


            item_score = neighbour_session.sum(0).A1

            top_k_item = np.argpartition(item_score, -self.k)[-self.k:]
            top_k_item = top_k_item[np.argsort(-item_score[top_k_item])]
            top_k_value = item_score[top_k_item]

            yield (top_k_item, top_k_value)
            continue

    def predict(self, session_items, session_timestamp):
        # initialize
        normalized = session_items.copy()
        normalized.data[normalized.data != 1] = 1
        self.current_session_sequence = session_items
        self.current_session = normalized

        session_length = session_items.max(axis=1).tocsr()
        session_items_cp = session_items.copy()

        if self.factor1 is True:
            nz = normalized.nonzero()
            session_items_cp[nz] -= session_length[nz[0]].transpose()
            session_items_cp.data = numpy.exp(session_items_cp.data / self.l1)
            self.current_session_weight_cache = session_items_cp
        else:
            self.current_session_weight_cache = normalized

        self.current_timestamp = session_timestamp

        neighbours = self.find_neighbours()
        scores = list(self.score_items(neighbours))
        return scores
