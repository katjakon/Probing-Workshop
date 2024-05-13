from collections import Counter, defaultdict
import random

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

class Dataset:
    def __init__(self) -> None:
        self.embeddings = np.array([[0.3, 0.5], [0.1, 0.9], [0.1, 0.9]])
        self.labels = [1, 0, 0]
        self.strings = ["a", "a", "a"]

class ClassifierProbe:
    
    def __init__(self, data_set) -> None:
        self.data_set = data_set
        self.probe = SGDClassifier()
    
    def fit(self):
        self.probe.fit(
            X=self.data_set.embeddings,
            y=self.data_set.labels
            )
    
    def predict(self, embeddings):
        return self.probe.predict(embeddings)


class ControlTaskProbe:

    def __init__(self, data_set):
        self.data_set = data_set
        self.n_labels = len(set(self.data_set.labels))
        self.probe = SGDClassifier()
        self.label_dict, self.control_task = self.create_control_task()

    def create_control_task(self):
        control_label_dict = dict()
        control_labels = []
        for string in self.data_set.strings:
            if string not in control_label_dict:
                rand_label = random.randrange(self.n_labels)
                control_label_dict[string] = rand_label
            control_labels.append(control_label_dict[string])
        return control_label_dict, control_labels
    
    def fit(self):
        self.probe.fit(
            X=self.data_set.embeddings,
            y=self.control_task
        )
    
    def predict(self, embeddings):
        preds = self.probe.predict(
            X=embeddings
        )
        return preds

class RandomProbe:

    def __init__(self, data_set) -> None:
        self.vocab = set(data_set.strings)
        self.data_set = data_set
        self.shape = data_set.embeddings[0].shape[0]
        self.probe = SGDClassifier()
        self.emb_dict, self.random_embeddings = self.create_rand_emb()
    
    def create_rand_emb(self):
        emb_dict = dict()
        random_embeddings = []
        for token in self.data_set.strings:
            print(token)
            # Create random embeddings for all tokens we haven't seen.
            if token not in emb_dict:
                emb_dict[token] = np.random.rand(self.shape)
            random_embeddings.append(
                emb_dict[token]
            )
        return emb_dict, random_embeddings
    
    def fit(self):
        self.probe.fit(X=self.random_embeddings, y=self.data_set.labels)
    
    def predict(self, strings):
        embeddings = []
        for token in strings:
            if token not in self.emb_dict:
                self.emb_dict[token] = np.random.rand(self.shape)
            embeddings.append(self.emb_dict[token])
        preds = self.probe.predict(X=embeddings)
        return preds
    

class MajorityBaseline:

    def __init__(self, data_set) -> None:
        self.data_set = data_set
        self.majority_dict = self.create_majority_dict(self.data_set)

    def create_majority_dict(self, data_set):
        counter_dict = defaultdict(Counter)
        for token, label in zip(data_set.strings, data_set.labels):
            counter_dict[token][label] += 1
        return {token: counter_dict[token].most_common(1)[0][0] for token in counter_dict.keys()}

    
if __name__ == "__main__":
    data = Dataset()
    clf_probe = MajorityBaseline(data_set=data)
    # clf_probe.fit()
    # test_embedding = [[0.1, 1]]
    # test_strings = ["a", "b"]
    # pred = clf_probe.predict(test_embedding)
    # print(pred)
