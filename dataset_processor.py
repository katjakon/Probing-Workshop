from datasets import Dataset, DatasetDict, load_dataset
from transformers import BertModel, BertTokenizerFast
from tqdm import tqdm
import torch

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
        data_column : str
            Column in the dataset that signifies the sentences/tokens
        labels_column : str
            Column in the dataset that signifies the labels
        dataset_type : str
            Type of data. Is either "tokens" or "sentences".
        """
        self.dataset = dataset

        self.model_name = model_name
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        # data columns
        self.data_column = data_column
        self.labels_column = labels_column

        self.dataset_keys = list(dataset.keys())
        self.label_tags = dataset[self.dataset_keys[0]].features[labels_column].feature.names

        self.create_probedict()

        assert dataset_type in self.DATASET_TYPES
        self.dataset_type = dataset_type

        # self.dataset_with_token_ids = dataset.map(lambda x: self._tokenize_and_align_labels(x), batched=True)
    
    @staticmethod
    def _align_labels_with_tokens(labels, word_ids, none_label=0):
        """
        Takes labels and word_ids to ali
        """
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

    def create_probedict(self):
        print("Creating dict for probing...")
        self.probedict = {k: {"embeddings": [], "labels": [], "ids": []} for k in self.dataset.keys()}
        for key in self.dataset.keys():
            print(f"Embeddings for {key}")
            data = self.dataset[key]
            tokens = data[self.data_column]
            labels = data[self.labels_column]

            for i, sentence in enumerate(tqdm(tokens)):
                sentence_ids, sentence_mapping = self.convert_to_ids(sentence)
                sentence_label_ids = labels[i]
                sentence_embeddings = self.get_word_representations(sentence_ids, sentence_mapping)

                # convert ids to string for readability
                sentence_tokens = self.convert_to_tokens(sentence_ids)
                sentence_labels = self.convert_to_labels(sentence_label_ids)

                self.probedict[key]["ids"].extend(sentence_tokens)
                self.probedict[key]["labels"].extend(sentence_labels)
                self.probedict[key]["embeddings"].extend(sentence_embeddings)

    def convert_to_tokens(self, ids):
        return [self.tokenizer.decode(ix) for ix in ids]

    def convert_to_labels(self, label_ids):
        return [self.label_tags[ix] for ix in label_ids]

    def convert_to_ids(self, string_list):
        word_id = 1
        mapping = []
        subtokens = []
        for string in string_list:
            token_subtokens = self.tokenizer.tokenize(string)
            subtokens.extend(token_subtokens)
            mapping.append((word_id, word_id+len(token_subtokens)-1))
            word_id += 1
        ids = self.tokenizer.convert_tokens_to_ids(subtokens)
        return ids, mapping

    def get_word_representations(self, ids, mapping, layer_idx=-1):
        ids = [101] + ids + [102]
        ids = torch.tensor([ids])

        output = self.model(ids, output_hidden_states=True)
        output = torch.squeeze(output[layer_idx][0], dim=0)
        # We don't need representations of [CLS] and [SEP]
        output = output[:-1]

        # taking the representation of the first subtoken
        token_repr = torch.zeros(size=(len(mapping), output.shape[-1]))
        for idx, (start, end) in enumerate(mapping):
            token_repr[idx] += output[start]
        return token_repr

if __name__ == "__main__":
    # pass
    data = load_dataset("universal_dependencies", "en_pronouns")
    bertmodel = "google-bert/bert-base-multilingual-cased"
    data_processor = DatasetProcessor(data, bertmodel, "tokens", "upos", dataset_type="tokens")
    # print(data_processor)