from __future__ import division
import numpy as np
import math
import utility

"""Inizializzazione e costruzione di Naive Bayes per la classificazione dei geni"""
class Naive_Bayes:
    def __init__(self, set):
        self.features_zero = {}
        self.features_one = {}
        self.struct = {}
        self.essential_length = utility.essential_length(set)
        self.set = np.delete(set, 0, 1)
        self.non_ess_length = self.set.shape[0] - self.essential_length
        self.num_of_features = self.set.shape[1]

    def train(self, X, y):
        transpose = np.transpose(self.set)
        for j in range(transpose.shape[0]):
            val = sorted(np.unique(transpose[j]))
            feature_values_zero = {}
            feature_values_one = {}

            for i in val:
                feature_values_zero[i] = 0
                feature_values_one[i] = 0
            self.features_zero[j] = feature_values_zero
            self.features_one[j] = feature_values_one
        self.struct[0] = self.features_zero
        self.struct[1] = self.features_one

        for j in range(self.num_of_features):
            for i in range(X.shape[0]):
                value = X[i, j]
                self.struct[y[i][0]][j][value] += 1

    def prints(self):
        for x in self.struct:
            print (x)
            for y in self.struct[x]:
                print " ", str(y)
                print "   ", str(self.struct[x][y])
                for k in self.struct[x][y]:
                    print "   ", str(k), ":" , str(self.struct[x][y][k])


    def classify(self, X_test, test_name):
        positive_label = math.log(self.essential_length/(self.essential_length + self.non_ess_length)) #P(y = 1)
        negative_label = math.log(self.non_ess_length/(self.non_ess_length + self.essential_length)) #P(y = 0)
        dict = {}
        for i in range(X_test.shape[0]):
            prob_of_ess = 1
            prob_of_non_ess = 1
            for j in range(X_test.shape[1]):
                prob_of_ess += math.log((self.struct[1][j][X_test[i][j]] + 1) / (self.essential_length + self.num_of_features))
                prob_of_non_ess += math.log((self.struct[0][j][X_test[i][j]] + 1) / (self.non_ess_length + self.num_of_features))
            prob_of_ess += positive_label
            prob_of_non_ess += negative_label
            prob_of_ess = math.exp(prob_of_ess)
            prob_of_non_ess = math.exp(prob_of_non_ess)
            prob_of_ess /= (prob_of_ess + prob_of_non_ess)
            dict[test_name[i][0]] = prob_of_ess
        return dict