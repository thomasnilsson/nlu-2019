import numpy as np
import argparse
import tensorflow as tf
import time
import os
import json
import pickle

import data_loader as dl
from models.model_list import get_model_class, get_model_info
import constants as C


def train_doc2vec(args):
    '''
    The doc2vec model doesn't fit the tensorflow
    template currently, so for the sake of time it has its
    own train function. It achieves around 60% accuracy
    with all the training data.
    '''

    # Set up file to save model
    model_type = "doc2vec"
    timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    if args.save_name:
        dir_name = args.save_name
    else:
        dir_name = "{}_{}".format(model_type, timestamp)

    model_save_dir = os.path.join(C.root_dir, C.model_dir, model_type, dir_name)
    saver_file = os.path.join(model_save_dir, C.doc2vec_model_fn)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Load data for model, and train it
    data = dl.load_data(loader=dl.load_doc2vec_data,
                        file_name=C.doc2vec_file,
                        train_limit=args.train_size_limit,
                        delete_file=args.delete_file)
    model = get_model_class(model_type)(data.train_single)

    # Train model
    model.train()

    # Predict and compute accuracy
    true_labels = data.validation["labels"]
    prediction = model.predict(data.validation["contexts"], data.validation["endings"])
    correct = (prediction == true_labels).sum()
    total = true_labels.shape[0]
    percentage = correct / total
    print("Validation Accuracy {}/{} ({:.2})".format(correct, total, percentage))

    # Save model with pickle
    pickle.dump(model, open(saver_file, "wb"))
    print("Doc2Vec model saved.")

def train():
    # Turn off some tf warnings
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument("--model_type", type=str, default="dummy",
                        help="model type to evaluate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--cell_size", type=int, default=128,
                        help="Cell state size for rnn-cells")
    parser.add_argument("--cell_height", type=int, default=1,
                        help="Height of cell stack for RNNs")
    parser.add_argument("--grad_clip_norm", type=float, default=10,
                        help="Value of norm to clip gradients at")
    parser.add_argument("--hidden_size", type=int, default=None,
                        help="Size to use for hidden layers in model")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="probability of dropout regularization")
    parser.add_argument("--save_freq", type=int, default=1,
                        help="How often tf model should be saved")
    parser.add_argument("--load_dirs", nargs="*", type=str, default=[],
                        help="Model directories to load from for use with combined model")
    parser.add_argument("--train_size_limit", type=int, default=None,
                        help="Size of single ending training set (omit for all 90k sentences)")
    parser.add_argument("--fine_tune", action="store_true", default=False,
                        help="Fine tune model end-to-end")
    parser.add_argument("--save_name", type=str, default=None,
                        help="Name of the subdirectory to save the model in.")
    parser.add_argument("--delete_file", type=bool, default=False,
                        help="Delete data file or not. (default = False)")

    args = parser.parse_args()

    if args.model_type == "doc2vec":
        train_doc2vec(args)
        return

    # Load data and embeddings
    data = dl.load_data(**get_model_info(args.model_type))

    word_to_id = data.word_to_id
    id_to_word = data.id_to_word # Vocab size ~33000
    embeddings = data.embeddings

    # Set up datasets to be used in training loop
    if data.train_single is not None:
      context_single = data.train_single["contexts"]
      end_single = data.train_single["endings"]
    context_multi = data.train_multi["contexts"]
    end_multi = data.train_multi["endings"]
    labels_multi = data.train_multi["labels"]

    context_val = data.validation["contexts"]
    end_val = data.validation["endings"]
    labels_val = data.validation["labels"]

    # Samples in training and validation data
    if data.train_single is not None:
      N_single = len(context_single)
      N_batches_single = np.ceil(N_single/args.batch_size).astype(int)

    N_multi = len(context_multi)
    N_val = len(context_val)
    N_batches_multi = np.ceil(N_multi/args.batch_size).astype(int)

    # Other model parameters
    model_config = {
        "learning_rate": args.lr,
        "cell_size": args.cell_size,
        "hidden_size": args.hidden_size,
        "grad_clip_norm": args.grad_clip_norm,
        "dropout": args.dropout,
        "cell_height": args.cell_height,
        "load_dirs": args.load_dirs,
        "fine_tune": args.fine_tune,
    }

    # For random operations
    randomizer = np.random.RandomState(seed=C.rand_seed)

    with tf.Session() as sess:
        # Set seed
        tf.random.set_random_seed(C.rand_seed)

        # Instantiate model
        model = get_model_class(args.model_type)(embeddings=embeddings, sess=sess,
                                                 word_to_id=word_to_id,
                                                 id_to_word=id_to_word, **model_config)
        assert model.train_single or model.train_multi, (
            "Selected model does not require training")

        if model.initialize:
            # Initialize all tf variables
            sess.run(tf.global_variables_initializer())

        # Create saving directory
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        if args.save_name:
            dir_name = args.save_name
        else:
            dir_name = "{}_{}".format(args.model_type, timestamp)
        model_save_dir = os.path.join(C.root_dir, C.model_dir, args.model_type, dir_name)
        saver_file = os.path.join(model_save_dir, args.model_type)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        saver = tf.train.Saver(max_to_keep=5)

        # Save model config
        config_file_path = os.path.join(model_save_dir, C.config_file)
        json.dump(model_config, open(config_file_path, "w"), indent=4)

        # Tensorboard summaries
        summary_dir = os.path.join(C.root_dir, C.tb_dir, dir_name)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        # Training loop
        print("Training")
        for epoch_i in range(args.epochs):
            # Train an epoch using single ending data
            if model.train_single:
                # Reset training loss
                train_single_loss = 0

                # One epoch
                for batch_i in range(N_batches_single):
                    batch_start = batch_i*args.batch_size
                    context_batch = context_single[
                        batch_start:batch_start+args.batch_size]
                    end_batch = end_single[batch_start:batch_start+args.batch_size]

                    # Take training step
                    batch_loss, global_step = model.train_step(
                            context_batch, end_batch, single=True, sess=sess,
                            summary_writer=summary_writer)
                    train_single_loss += batch_loss

            # Train an epoch using multi ending data
            if model.train_multi:
                # Reset training loss
                train_multi_loss = 0

                # One epoch
                for batch_i in range(N_batches_multi):
                    batch_start = batch_i*args.batch_size
                    context_batch = context_multi[
                        batch_start:batch_start+args.batch_size]
                    end_batch = end_multi[batch_start:batch_start+args.batch_size]
                    labels_batch = labels_multi[batch_start:batch_start+args.batch_size]

                    # Take training step
                    batch_loss, global_step = model.train_step(
                            context_batch, end_batch, single=False,
                            labels=labels_batch, sess=sess,
                            summary_writer=summary_writer)
                    train_multi_loss += batch_loss

            # Validate
            prediction = model.predict(context_val, end_val,
                    sess=sess, summary_writer=summary_writer)
            acc_val = np.sum((prediction == labels_val).astype(int))/N_val

            # Report at end of epoch
            msg = "End of epoch {}".format(epoch_i)

            if model.train_single:
                single_epoch_loss = train_single_loss / N_batches_single
                msg += " - train_single_loss: {:0.3f}".format(single_epoch_loss)
            if model.train_multi:
                multi_epoch_loss = train_multi_loss / N_batches_multi
                msg += " - train_multi_loss: {:0.3f}".format(multi_epoch_loss)

            msg += " - validation_accuracy: {:0.3f}".format(acc_val)
            print(msg)

            # Saving
            if epoch_i % args.save_freq == 0:
                # Save model
                saver.save(sess, saver_file, global_step=global_step)


if __name__ == "__main__":
    train()
