from collections import Counter, defaultdict
import random
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

class Dataset:

    def __init__(self) -> None:
        self.probing = {
            "train": {
                "ids": [0, 1, 2, 3, 3, 3],
                "labels": ["a", "b", "c", "d", "d", "e"],
                "embeddings": np.array([
                    [0, 1, 2],
                    [0, 1, 1],
                    [0, 2, 2],
                    [0, 3, 3],
                    [0, 3, 2], 
                    [0, 3, 1]
                ])
                      },

        }
    
    def __getitem__(self, key):
        return self.probing[key]

class ClassifierProbe:
    
    def __init__(self, data_set, clf, clf_kwargs: dict=None) -> None:
        """Initialize a probing classifier.

        Args:
            data_set (Custom Dataset type): Should have attributes for embeddings, labels & strings.
            clf (scikit-learn classifier): For instance, SGDClassifier or MLPClassifier
            clf_kwargs (dict): Keyword arguments to be given to clf
        """
        self.data_set = data_set
        if clf_kwargs is None:
            clf_kwargs = dict()
        self.probe = clf(**clf_kwargs)
    
    def fit(self):
        """Fit the given probe to the given classifier."""
        self.probe.fit(
            X=self.data_set["train"]["embeddings"],
            y=self.data_set["train"]["labels"]
            )
    
    def predict(self, embeddings):
        """Predict given instances.

        Args:
            embeddings (matrix-like): Predict labels based on given embeddings.

        Returns:
            1-d array: Predicted labels.
        """
        return self.probe.predict(embeddings)


class ControlTaskProbe:

    def __init__(self, data_set, clf, clf_kwargs: dict=None):
        """Initialize a control tasks probe.

        Args:
            data_set (Custom Dataset type): Should have attributes for embeddings, labels & strings.
            clf (scikit-learn classifier): For instance, SGDClassifier or MLPClassifier
            clf_kwargs (dict): Keyword arguments to be given to clf
    """
        self.data_set = data_set
        self.n_labels = len(set(self.data_set["train"]["labels"]))
        if clf_kwargs is None:
            clf_kwargs = dict()
        self.probe = clf(**clf_kwargs)
        self.label_dict, self.control_task = self.create_control_task()

    def create_control_task(self):
        """Create the control tasks with random labels.

        Returns:
            (dict, list[int]): 2-tuple. Dictionary maps tokens to their control label, list of ints are the control labels mapped to the original data set.
        """
        control_label_dict = dict()
        control_labels = []
        for string in self.data_set["train"]["ids"]:
            if string not in control_label_dict:
                rand_label = random.randrange(self.n_labels)
                control_label_dict[string] = rand_label
            control_labels.append(control_label_dict[string])
        return control_label_dict, control_labels
    
    def fit(self):
        """Fit the given probe to the given classifier."""
        self.probe.fit(
            X=self.data_set["train"]["embeddings"],
            y=self.control_task
        )
    
    def predict(self, embeddings):
        """Predict given instances.

        Args:
            embeddings (matrix-like): Predict labels based on given embeddings.

        Returns:
            1-d array: Predicted labels.
        """
        preds = self.probe.predict(
            X=embeddings
        )
        return preds

class RandomProbe:

    def __init__(self, data_set, clf, clf_kwargs: dict=None) -> None:
        """Initialize a probe with random embeddings.

        Args:
            data_set (Custom Dataset type): Should have attributes for embeddings, labels & strings.
            clf (scikit-learn classifier): For instance, SGDClassifier or MLPClassifier
            clf_kwargs (dict): Keyword arguments to be given to clf
    """
        self.vocab = set(data_set["train"]["ids"])
        self.data_set = data_set
        self.shape = data_set["train"]["embeddings"][0].shape[0]
        if clf_kwargs is None:
            clf_kwargs = dict()
        self.probe = clf(**clf_kwargs)
        self.emb_dict, self.random_embeddings = self.create_rand_emb()
    

    def create_rand_emb(self):
        """Create random embeddings for word types.

        Returns:
            (dict, matrix-like): 2-tuple, contains dictionary with token types and their respective embeedings and the data set but with random embeddings.
        """
        emb_dict = dict()
        random_embeddings = []
        for token in self.data_set["train"]["ids"]:
            # Create random embeddings for all tokens we haven't seen.
            if token not in emb_dict:
                emb_dict[token] = np.random.rand(self.shape)
            random_embeddings.append(
                emb_dict[token]
            )
        return emb_dict, random_embeddings
    
    def fit(self):
        """Fit the given probe to the given data set"""
        self.probe.fit(X=self.random_embeddings, y=self.data_set["train"]["labels"])
    
    def predict(self, token_ids: list[int]):
        """Predict given instances.

        Args:
            instances list[str]: Predict labels based on given token strings.

        Returns:
            1-d array: Predicted labels.
        """
        X = []
        for token in token_ids:
            if token not in self.emb_dict:
                self.emb_dict[token] = np.random.rand(self.shape)
            X.append(self.emb_dict[token])
        preds = self.probe.predict(X=X)
        return preds
    

class MajorityBaseline:

    def __init__(self, data_set) -> None:
        """Initialize the majority baseline.

        Args:
            data_set (Custom Dataset type): Should have attributes for embeddings, labels & strings.
        """
        self.data_set = data_set
        self.most_common_overall, self.majority_dict = None, dict()

    def create_majority_dict(self, data_set):
        """Count up labels and assign majority tags.

        Args:
            data_set (Custom Dataset type): Should have attributes for embeddings, labels & strings

        Returns:
            (str, dict): 2-tuple, string that represents most common label in given data set and dictionary which contains the most common label for each token.
        """
        counter_dict = defaultdict(Counter)
        all_counts = Counter()
        for token, label in zip(data_set["train"]["ids"], data_set["train"]["labels"]):
            counter_dict[token][label] += 1
            all_counts[label] += 1
        most_common_overall = max(all_counts, key=lambda x: all_counts[x])
        return most_common_overall, {token: counter_dict[token].most_common(1)[0][0] for token in counter_dict.keys()}
    
    def fit(self):
        """Fit the given probe to the given data set"""
        self.most_common_overall, self.majority_dict = self.create_majority_dict(self.data_set)

    def predict(self, token_ids):
        """Predict given instances.

        Args:
            token_ids list[int]: Token ids to predict.

        Returns:
            1-d array: Predicted labels.
        """
        preds = []
        for string in token_ids:
            pred = self.majority_dict.get(string, self.most_common_overall)
            preds.append(pred)
        return preds


if __name__ == "__main__":
    data = Dataset()
    clf_probe = MajorityBaseline(data_set=data)
    clf_probe.fit()
    test_embeds = np.array([[0, 1, 2], [0, 3, 0]])
    test_ids = [0, 3]
    test_labels = ["a", "e"]
    preds = clf_probe.predict(test_ids)
    print(preds)
