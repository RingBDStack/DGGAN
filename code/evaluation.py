import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LinkPrediction():
    def __init__(self, config):
        self.links = [[], [], []]
        sufs = ['_0', '_50', '_100']
        for i, suf in enumerate(sufs):
            with open(config.test_file + suf) as infile:
                for line in infile.readlines():
                    s, t, label = [int(item) for item in line.strip().split()]
                    self.links[i].append([s, t, label])

    def evaluate(self, embedding_matrix):
        test_y = [[], [], []]
        pred_y = [[], [], []]
        pred_label = [[], [], []]
        for i in range(len(self.links)):
            for s, t, label in self.links[i]:
                test_y[i].append(label)
                pred_y[i].append(embedding_matrix[0][s].dot(embedding_matrix[1][t]))
                if pred_y[i][-1] >= 0:
                    pred_label[i].append(1)
                else:
                    pred_label[i].append(0)

        auc = [0, 0, 0]
        for i in range(len(test_y)):
            auc[i] = roc_auc_score(test_y[i], pred_y[i])
        return auc
