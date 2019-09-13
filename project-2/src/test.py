import numpy as np
import argparse
import os
import tensorflow as tf
import json
import pickle

import constants as C
import data_loader as dl
from models.model_list import get_model_class, get_model_info

def eval_doc2vec(model_path):
    if not model_path:
        print("ERROR: No model path defined, couldn't test model!")
        return

    model_type = "doc2vec"
    model = pickle.load(open(model_path, "rb"))

    data = dl.load_doc2vec_data(load_single_endings=False)
    test = data.test
    true_labels = test["labels"]

    # Predict and calculate
    prediction = model.predict(test["contexts"], test["endings"])
    correct = (prediction == true_labels).sum()
    total = true_labels.shape[0]
    percentage = correct / total

    kwargs = {
        "epochs" : model.epochs,
        "embedding_size" : model.vector_size
    }

    print("")
    print("Validation: {}".format(model_type))
    print("----------------------------")
    print("Options:")
    for k in kwargs:
        print("{}: {}".format(k, kwargs[k]))
    print("----------------------------")
    print("Accuracy {}/{} ({:.2})".format(correct, total, percentage))
    print("----------------------------")


def predict_batched(model, sess, contexts, endings, batch_size):
        N_tests = len(contexts)
        N_batches= np.ceil(N_tests/batch_size).astype(int)

        prediction = np.array([])
        for batch_i in range(N_batches):
            batch_start = batch_i*batch_size
            context_batch = contexts[
                    batch_start:batch_start+batch_size]
            end_batch = endings[batch_start:batch_start+batch_size]

            prediction_batch = model.predict(context_batch, end_batch, sess=sess)
            prediction = np.concatenate((prediction, prediction_batch))

        return prediction

def eval():
    # Turn off some tf warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser(description='Evaluate model')

    parser.add_argument("--model_type", type=str, default="dummy",
                        help="Model type to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--load", type=str, default=None,
                        help="Directory to load tf model checkpoint from")

    args = parser.parse_args()

    if args.model_type == "doc2vec":
        eval_doc2vec(args.load)
        return

    # Load data
    data = dl.load_data(**get_model_info(args.model_type))
    embeddings = data.embeddings
    word_to_id = data.word_to_id
    id_to_word = data.id_to_word
    test = data.test
    test_no_lab = data.test_wo_lab

    # Additional arguments
    kwargs = {}

    # Load model config
    if args.load:
        config_file_path = os.path.join(args.load, C.config_file)
        model_config = json.load(open(config_file_path))
        kwargs.update(model_config)  # Add arguments from model config

    with tf.Session() as sess:
        # Set seed
        tf.set_random_seed(C.rand_seed)

        # Instantiate model
        model = get_model_class(args.model_type)(embeddings=embeddings,
                                                 word_to_id=word_to_id,
                                                 id_to_word=id_to_word,
                                                 sess=sess, **kwargs)
        assert not (model.train_single or model.train_multi)\
            or bool(args.load), "Model requires saved weights"

        if args.load:
            # Load tf model
            saver = tf.train.Saver()
            cp_file = tf.train.latest_checkpoint(args.load)
            saver.restore(sess, cp_file)

        # Validate on test
        print("Running validation...")
        test_prediction = predict_batched(model, sess, test["contexts"], test["endings"],
                args.batch_size)
        true_labels = test["labels"]
        correct = np.sum((test_prediction == true_labels).astype(int))
        total = true_labels.shape[0]
        percentage = correct/total

        try:
          # Output prediction to csv file
          prediction = predict_batched(model, sess, test_no_lab["contexts"],
                  test_no_lab["endings"], args.batch_size)
          prediction_save_path = os.path.join(C.root_dir, C.data_dir, C.prediction_file)
          np.savetxt(prediction_save_path, prediction + 1, fmt='%i', delimiter='\n')
          print("saved predictions to {}".format(prediction_save_path))
        except TypeError:
          print("no test data wo. labels found")

    print("")
    print("Validation: {}".format(args.model_type))
    print("----------------------------")
    print("Options:")
    for k in kwargs:
        print("{}: {}".format(k, kwargs[k]))
    print("----------------------------")
    print("Accuracy {}/{} ({:.2})".format(correct, total, percentage))
    print("----------------------------")


if __name__ == "__main__":
    eval()
