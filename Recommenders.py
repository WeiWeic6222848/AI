import math
from collections import Counter

import numpy
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse, spatial
from scipy.sparse import hstack
from _operator import itemgetter


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
        self.session = session
        self.session_timestamp = session_timestamp
        return self

    def possible_neighbour_sessions(self, session_items):
        # any session with at least one common item as the one in the testing session is considered as the possible neighbour.
        # create a mapping of prediction sessions to possible neighboring sessions
        neighbours = session_items * self.session.transpose()
        neighbours.data[neighbours.data != 1] = 1  # set all value to 1
        return neighbours

    def find_neighbours(self, session_items):
        # neighbour candidate
        possible_neighbours = self.possible_neighbour_sessions(session_items)
        # get top k according to similarity
        possible_neighbours = self.cal_similarity(session_items,
                                                  possible_neighbours)

        top_neighbors = (-possible_neighbours.data).argsort()[:self.kneighbor]
        possible_neighbours.col = possible_neighbours.col[top_neighbors]
        possible_neighbours.row = possible_neighbours.row[top_neighbors]
        possible_neighbours.data = possible_neighbours.data[top_neighbors]

        return possible_neighbours

    # calculate similarity between target session and possible neighbors
    def cal_similarity(self, session_items, session_to_neighbors_matrix):

        for session_to_neighbor in session_to_neighbors_matrix:
            neighbour_session_items = self.session.multiply(session_to_neighbor.transpose())
            similarity = self.cosine_similarity(neighbour_session_items,
                                                session_items)
            if self.factor2 is True:
                similarity = similarity.multiply(numpy.exp(
                    (self.session_timestamp.data - self.current_timestamp) / self.l2))

            return similarity

    # cosine similarity of neighbor sessions against one current session
    def cosine_similarity(self, s1, s2):
        similarity = self.current_session_weight_cache * s1.transpose()
        sessions = list(s1.nonzero())
        sessions_length = Counter(sessions[0]).most_common()
        for col, length in sessions_length:
            similarity[0, col] /= math.sqrt(length * len(s2.data))
        return similarity

    # scoring item according to their session
    def score_items(self, neighbours, current_session, sequence_information):
        scores = {}
        sessions = self.session.multiply(neighbours.transpose())
        neighbours = neighbours.tocsc()

        for session in sessions.nonzero()[0]:
            items = self.session.col[self.session.row == session]
            common_items_idx = numpy.nonzero(numpy.in1d(items, sequence_information))[-1][0]
            similarity = neighbours.getcol(session).data[0]
            for idx, item in enumerate(items):
                old_score = scores.get(item)
                if self.factor3 is True:
                    new_score = math.exp(
                        -math.fabs(idx - common_items_idx) / self.l3) * \
                               similarity
                else:
                    new_score = similarity

                if old_score is None:
                    scores.update({item: new_score})
                else:
                    new_score += old_score
                    scores.update({item: new_score})
        return scores

    def predict(self, session_items, session_timestamp, sequence_information,
                k=20):
        # initialize
        self.current_session_weight_cache = scipy.sparse.csr_matrix(session_items.get_shape())

        sessions = list(session_items.nonzero())
        sessions_length = Counter(sessions[0]).most_common()

        for row, length in sessions_length:
            cols = session_items.getrow(row).nonzero()[1]
            for idx, col in enumerate(cols):
                if self.factor1 is True:
                    weight = math.exp((idx + 1 - length) / self.l1)
                else:
                    weight = 1
                # if there is several identical items in one session, choose the biggest weight as the item's weight
                if self.current_session_weight_cache[row, col] != 0:
                    self.current_session_weight_cache[row, col] = max(weight, self.current_session_weight_cache[row, col])
                else:
                    self.current_session_weight_cache[row, col] = weight

        self.current_timestamp = session_timestamp.data[0]

        neighbours = self.find_neighbours(session_items)
        scores = self.score_items(neighbours, session_items,
                                  sequence_information)
        scores_sorted_list = sorted(scores.items(), key=lambda x: x[1],
                                    reverse=True)[:k]
        return scores_sorted_list
