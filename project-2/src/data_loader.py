import numpy as np
import collections
import os
import nltk.tokenize as tokenizer
import pickle
import pandas as pd
from gensim.models.doc2vec import TaggedDocument

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

# Local modules
import constants as C
from load_embedding import numberbatch_embedding

val_path = os.path.join(C.root_dir, C.data_dir, C.val_file)
train_path = os.path.join(C.root_dir, C.data_dir, C.train_file)
test_path_no_lab = os.path.join(C.root_dir, C.data_dir, C.test_file_unlabelled)
test_path_lab = os.path.join(C.root_dir, C.data_dir, C.test_file_labelled)

empty_data = {"contexts": [0], "endings": [0], "labels": [0]},  # Just dummy placeholder

class StoryClozeData(object):
    def __init__(self, word_to_id, id_to_word, embeddings, train_single,
                 train_multi, validation, test_w_lab, test_wo_lab):

        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.embeddings = embeddings
        self.train_single = train_single
        self.train_multi = train_multi
        self.validation = validation
        self.test = test_w_lab
        self.test_wo_lab = test_wo_lab


def tokenize(sentences):
    """
    Uses nltk to tokenize sentences
    """

    res = []
    for row in sentences:
        # Strip period token from each sentence
        res.append(
            [tokenizer.word_tokenize(sent.lower())[:-1] for sent in row])
    return res


def indexize(sentences, word_to_id):
    """
    Turns tokenized sentences into indexes of vocabulary
    """

    def lookup_w2id(tok):
        if tok in word_to_id:
            return word_to_id[tok]
        else:
            return word_to_id[C.unk_token]

    res = []
    for row in sentences:
        res.append([[lookup_w2id(tok) for tok in sent] for sent in row])
    return res


def load_set(data_path,
             data_type,
             train_limit=False,
             do_tokenize=True):
    # Read with Pandas
    csv_data = pd.read_csv(data_path)

    storyid_col = 0

    # Intitialize dataset as empty dict
    dataset = {}

    # Extract story ids as array
    dataset["storyids"] = np.array(csv_data.iloc[:, storyid_col])

    # The two datasets have different indexing,
    # use if-else to handle this
    if data_type == "validation":
        # Shuffle to prepare for splitting
        csv_data = csv_data.sample(frac=1.0, random_state=C.rand_seed)

        context_cols = [
            "InputSentence1", "InputSentence2", "InputSentence3",
            "InputSentence4"
        ]
        ending_cols = ["RandomFifthSentenceQuiz1", "RandomFifthSentenceQuiz2"]
        label_col = ["AnswerRightEnding"]

        # Extract first 4 sentence as matrix (context)
        dataset["contexts"] = np.array(csv_data[context_cols])

        # Extract all endings (both wrong and correct)
        dataset["endings"] = np.array(csv_data[ending_cols])

        # Extract labels (binary), indicating the correct answer.
        # Make them zero indexed
        dataset["labels"] = np.array(csv_data[label_col]).squeeze() - 1

    elif data_type == "test":
        # Test data doesn't have labels
        # Shuffle to prepare for splitting
        csv_data = csv_data.sample(frac=1.0, random_state=C.rand_seed)

        context_cols = [
            "InputSentence1", "InputSentence2", "InputSentence3",
            "InputSentence4"
        ]
        ending_cols = ["RandomFifthSentenceQuiz1", "RandomFifthSentenceQuiz2"]

        # Extract first 4 sentence as matrix (context)
        dataset["contexts"] = np.array(csv_data[context_cols])

        # Extract all endings (both wrong and correct)
        dataset["endings"] = np.array(csv_data[ending_cols])


    elif data_type == "train":
        context_cols = ["sentence" + str(k) for k in range(1, 5)]
        ending_col = ["sentence5"]

        # Limits the training set to the first "train_limit" rows
        if train_limit:
            csv_data = csv_data[:train_limit]

        # Extract first 4 sentence as matrix (context)
        dataset["contexts"] = np.array(csv_data[context_cols])

        # Extract the correct endings
        dataset["endings"] = np.array(csv_data[ending_col])

    # Tokenize sentences and endings
    if do_tokenize:
        dataset["contexts"] = tokenize(dataset["contexts"])
        dataset["endings"] = tokenize(dataset["endings"])

    return dataset


def split_validation(val_data):
    # Split validation set in 2
    val_len = len(val_data["endings"])
    val_samples = int(C.validation_frac * val_len)

    train_multi = {}
    validation = {}
    for k in val_data:
        validation[k] = val_data[k][:val_samples-1]
        train_multi[k] = val_data[k][val_samples:]

    return train_multi, validation


def load_preprocess(train_limit=False):

    val_data = load_set(val_path, data_type="validation")
    test_data_wo_lab = load_set(test_path_no_lab, data_type="test")
    train_data = load_set(train_path, data_type="train", train_limit=train_limit)

    # Treat downloaded test data as our test data with labels
    test_data_w_lab = load_set(test_path_lab, data_type="validation")

    def conc(x):
        return sum([sum(r, []) for r in x], [])

    word_list = conc(train_data["contexts"]) + conc(train_data["endings"])

    # Count appearances of each word
    counter = collections.Counter(word_list)
    word_list = [w[0] for w in counter.most_common(C.vocab_size)] + [C.unk_token]

    # Create vocabulary
    id_to_word = list(set(word_list))
    vocab_size = len(id_to_word)
    word_to_id = dict(zip(id_to_word, list(range(vocab_size))))

    # Change tokens to word indexes
    train_single = {}
    train_single["contexts"] = indexize(train_data["contexts"], word_to_id)
    train_single["endings"] = indexize(train_data["endings"], word_to_id)

    val_data["contexts"] = indexize(val_data["contexts"], word_to_id)
    val_data["endings"] = indexize(val_data["endings"], word_to_id)

    test_data_wo_lab["contexts"] = indexize(test_data_wo_lab["contexts"], word_to_id)
    test_data_wo_lab["endings"] = indexize(test_data_wo_lab["endings"], word_to_id)

    test_data_w_lab["contexts"] = indexize(test_data_w_lab["contexts"], word_to_id)
    test_data_w_lab["endings"] = indexize(test_data_w_lab["endings"], word_to_id)

    # Set up embedding vectors
    embedding_path = os.path.join(C.root_dir, C.data_dir, C.embeddings_file)

    embeddings = numberbatch_embedding(word_to_id, embedding_path,
                                       C.embed_dim, vocab_size)

    # Split validation set in 2
    train_multi, validation = split_validation(val_data)

    pp_data = StoryClozeData(word_to_id, id_to_word, embeddings, train_single,
                             train_multi, validation, test_data_w_lab, test_data_wo_lab)

    return pp_data


def load_data(train_limit=False, loader=load_preprocess, file_name=C.preprocessed_file,
              delete_file=False):
    pp_path = os.path.join(C.root_dir, C.data_dir, file_name)

    # This tag is used for forcing generation of a new data file
    if delete_file:
        try:
            os.remove(pp_path)
            print("Deleted the pre-existing data file")
        except:
            print("No pre-existing file could be deleted")

    try:
        print("Trying to load preprocessed data from {}".format(pp_path))
        pp_data = pickle.load(open(pp_path, "rb"))

        print("Loading successful!")
    except:
        print("Loading failed, preprocessing data...")
        pp_data = loader(train_limit=train_limit)

        print("Preprocessing done!")

        pickle.dump(pp_data, open(pp_path, "wb"))
        print("Preprocessed data saved to {}".format(pp_path))
    return pp_data


"""
SENTIMENT DATA BELOW
"""

def extract_sentiment(stories):

    analyser = SentimentIntensityAnalyzer()
    out = []

    for story in stories:
        story_sentiments = []

        for s in story:
            sentiment = list(analyser.polarity_scores(s).values())[0:2]
            story_sentiments.append(sentiment)

        out.append(story_sentiments)

    return out


def load_sentiment(train_limit):

    val_data = load_set(val_path, data_type="validation", do_tokenize=False)
    train_data = load_set(train_path,
                          data_type="train",
                          train_limit=train_limit,
                          do_tokenize=False)
    test_data_w_lab = load_set(test_path_lab,
                               data_type="validation",
                               do_tokenize=False)
    test_data_wo_lab = load_set(test_path_no_lab,
                                data_type="test",
                                do_tokenize=False)

    train_single = {}
    train_single["contexts"] = extract_sentiment(train_data["contexts"])
    train_single["endings"] = extract_sentiment(train_data["endings"])

    test_data_w_lab["contexts"] = extract_sentiment(test_data_w_lab["contexts"])
    test_data_w_lab["endings"] = extract_sentiment(test_data_w_lab["endings"])

    test_data_wo_lab["contexts"] = extract_sentiment(test_data_wo_lab["contexts"])
    test_data_wo_lab["endings"] = extract_sentiment(test_data_wo_lab["endings"])

    val_data["contexts"] = extract_sentiment(val_data["contexts"])
    val_data["endings"] = extract_sentiment(val_data["endings"])

    # Split validation set in 3
    train_multi, validation = split_validation(val_data)

    pp_data = StoryClozeData(
        None,  # word_to_id
        None,  # id_to_word
        None,  # embeddings
        train_single,
        train_multi,
        validation,
        test_data_w_lab,
        test_data_wo_lab)

    return pp_data


def load_combined(train_limit):
    """
    returns data on format
    StoryClozeData -> attribute (e.g. train_single) ->
    dict with keys "contexts", "endings", ("labels", unchanged), where each of
    these is a -> list of samples, and each sample is a ->
    dict with filenames (e.g. C.preprocessed_file or C.sentiment_file) as
    keys, and values -> original sample preprocessed using config related to filename key

    """
    # Load all needed preprocessed data
    load_configs = [
        (load_preprocess, C.preprocessed_file),
        (load_sentiment, C.sentiment_file),
        (load_doc2vec_data, C.doc2vec_file)  # ??
    ]

    pp_datasets = {cfg[1]: load_data(train_limit, cfg[0], cfg[1]) for cfg in load_configs}

    # One layer dict
    word_to_id = {k: pp_datasets[k].word_to_id for k in pp_datasets}
    id_to_word = {k: pp_datasets[k].id_to_word for k in pp_datasets}
    embeddings = {k: pp_datasets[k].embeddings for k in pp_datasets}

    # Make each entry dict, to allow for batching
    dk = C.preprocessed_file

    attributes = ["train_multi", "validation", "test", "test_wo_lab"]
    new_splits = {}
    for attr in attributes:
        attribute_length = len(getattr(pp_datasets[dk], attr)["contexts"])
        contexts = [
            {k: getattr(pp_datasets[k], attr)["contexts"][i] for k in pp_datasets}
            for i in range(attribute_length)
        ]
        endings = [
            {k: getattr(pp_datasets[k], attr)["endings"][i] for k in pp_datasets}
            for i in range(attribute_length)
        ]

        new_splits[attr] = {
            "contexts": contexts,
            "endings": endings
        }

        if attr != "train_single" and attr != "test_wo_lab":
            new_splits[attr]["labels"] = getattr(pp_datasets[dk], attr)["labels"]


    pp_data = StoryClozeData(
        word_to_id,
        id_to_word,
        embeddings,
        None,
        new_splits["train_multi"],
        new_splits["validation"],
        new_splits["test"],
        new_splits["test_wo_lab"])

    return pp_data


"""
DOC2VEC DATA BELOW
"""

def load_doc2vec_data(load_single_endings=True,
                      stem=True,
                      train_limit=None):
    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))

    # String to list of stemmed words
    def get_norm_words(string):
        if stem:
            return [
                stemmer.stem(word)
                for word in tokenizer.tokenize(string.lower())
                if not word in stop_words
            ]

        else:
            return tokenizer.tokenize(string)

    # List<List<string>> to String (stemmed)
    def flatten_sentences(sentence_list):
        normed_sent_list = [get_norm_words(sen) for sen in sentence_list]
        flat_word_list = [
            val for sublist in normed_sent_list for val in sublist
        ]
        return flat_word_list

    # Can be used to create a dataset (train or test - or val, doesnt matter)
    def build_dataset(stories_df, sent_cols, tagged_doc=False):
        docs = []
        if tagged_doc:
            titles = list(stories_df['storytitle'])
        for i in range(len(stories_df)):
            sentences = list(stories_df[sent_cols].iloc[i])

            if tagged_doc:
                all_sents = flatten_sentences(sentences)
                title = get_norm_words(titles[i])
                doc = TaggedDocument(all_sents, tags=title)
            else:
                doc = flatten_sentences(sentences)
            docs.append(doc)
        return docs

    train_stories = pd.read_csv(train_path)
    val_stories = pd.read_csv(val_path)
    test_stories = pd.read_csv(test_path_lab)
    test_no_lab_stories = pd.read_csv(test_path_no_lab)

    #### LOAD SINGLE ENDINGS FOR DOC2VEC #############
    if load_single_endings:
        # Use train limit to limit the number of train stories to load
        # If no limit is specified, load everything
        if not train_limit:
            train_limit = len(train_stories)

        print("Loading Doc2Vec train set (size=%i)..." % train_limit)
        train_data = build_dataset(
            train_stories.iloc[:train_limit],
            ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"],
            tagged_doc=True)
    else:
        print("Omitted loading Doc2Vec train set!")
        train_data = None
    ##################################################

    print("Loading Doc2Vec val set...")
    val_contexts = build_dataset(val_stories, [
        "InputSentence1", "InputSentence2", "InputSentence3", "InputSentence4"
    ])
    # Process val endings separately, and then zip them
    val_endings_1 = build_dataset(val_stories, ["RandomFifthSentenceQuiz1"])
    val_endings_2 = build_dataset(val_stories, ["RandomFifthSentenceQuiz2"])
    val_endings = list(zip(val_endings_1, val_endings_2))

    # Zero index the endings
    val_labels = np.array(val_stories.AnswerRightEnding) - 1

    val_data = {
        "contexts": val_contexts,
        "endings": val_endings,
        "labels": val_labels
    }

    # Process test data set with labels
    test_contexts = build_dataset(test_stories, [
        "InputSentence1", "InputSentence2", "InputSentence3", "InputSentence4"
    ])


    # Process val endings separately, and then zip them
    test_endings_1 = build_dataset(test_stories, ["RandomFifthSentenceQuiz1"])
    test_endings_2 = build_dataset(test_stories, ["RandomFifthSentenceQuiz2"])
    test_endings = list(zip(test_endings_1, test_endings_2))

    # Zero index the endings
    test_labels = np.array(test_stories.AnswerRightEnding) - 1

    test_data = {
        "contexts": test_contexts,
        "endings": test_endings,
        "labels": test_labels
    }

    # Process test data set with labels
    test_no_lab_contexts = build_dataset(test_no_lab_stories, [
        "InputSentence1", "InputSentence2", "InputSentence3", "InputSentence4"
    ])


    # Process val endings separately, and then zip them
    test_no_lab_endings_1 = build_dataset(test_no_lab_stories, ["RandomFifthSentenceQuiz1"])
    test_no_lab_endings_2 = build_dataset(test_no_lab_stories, ["RandomFifthSentenceQuiz2"])
    test_no_lab_endings = list(zip(test_no_lab_endings_1, test_no_lab_endings_2))

    # Zero index the endings
    test_no_lab_labels = [0]

    test_no_lab_data = {
        "contexts": test_no_lab_contexts,
        "endings": test_no_lab_endings,
        "labels": test_no_lab_labels
    }

    # Split validation set in 3
    train_multi, validation = split_validation(val_data)

    pp_data = StoryClozeData(
        None,  # word_to_id
        None,  # id_to_word
        None,  # embeddings
        train_data,
        train_multi,
        validation,
        test_data,
        test_no_lab_data)

    return pp_data
