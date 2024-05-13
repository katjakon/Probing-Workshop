from datasets import Dataset, DatasetDict, load_dataset
from transformers import BertPreTrainedModel, BertTokenizerFast

class DatasetProcessor:
    """
    Represents a given labeled dataset in a generalized form.
    """

    DATASET_TYPES = ["sentences", "tokens"]

    def __init__(self, dataset, model_name, data_column, labels_column, dataset_type="tokens"):
        """
        Initialization

        Parameters
        __________
        dataset : Dataset
            The dataset
        model_name : str
            The name of the model we use to embed and tokenise the dataset.
            Must be possible to load with BertPreTrainedModel.
        # TODO: write the rest
        """
        self.dataset = dataset
        self.model_name = model_name
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertPreTrainedModel.from_pretrained(model_name)

        # data columns
        self.data_column = data_column
        self.labels_column = labels_column

        assert dataset_type in self.DATASET_TYPES
        self.dataset_type = dataset_type
    
    @staticmethod
    def _align_labels_with_tokens(labels, word_ids, none_label=0):
        print(labels)
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word
                current_word = word_id
                label = none_label if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                new_labels.append(label)

        return new_labels


    def _tokenize_and_align_labels(self, examples):
        # TODO: make it so that the object has both tokens and labels after this
        assert self.dataset_type == "tokens"

        tokenized_inputs = self.tokenizer(
            examples[self.data_column], truncation=True, is_split_into_words=True
        )
        all_labels = examples[self.labels_column]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))
        tokenized_inputs["labels"] = new_labels

        return tokenized_inputs

    def embed(self):
        """
        Add the embeddings column to the dataset
        """
        pass

    def labeled_list(self):
        pass

if __name__ == "__main__":
    data = load_dataset("universal_dependencies", "en_pronouns")
    bertmodel = "google-bert/bert-base-multilingual-cased"
    data_processor = DatasetProcessor(data, bertmodel, "tokens", "upos", dataset_type="tokens")
    print(data_processor)