import tensorflow as tf
import numpy as np

from models.model import Model
import constants as C

class RNNLanguageModel(Model):
    def __init__(self, id_to_word, learning_rate=1e-3,grad_clip_norm=10,
            cell_size=128, cell_height=1, **kwargs):
        super(RNNLanguageModel, self).__init__(**kwargs)

        self.train_single = True

        self.vocab_size = len(id_to_word)

        # Global step
        self.global_step = tf.train.get_or_create_global_step()

        # Placeholders
        self.batch_size = tf.placeholder(dtype=tf.int32, name="batch_size", shape=())
        self.texts = tf.placeholder(dtype=tf.int32, name="texts", shape=(None, None))
        self.text_lengths = tf.placeholder(dtype=tf.int32, name="text_lengths",
                shape=(None,))
        self.end_lengths = tf.placeholder(dtype=tf.int32, name="end_lengths",
                shape=(None,))

        self.rnn_inputs = self.texts[:, :-1]
        self.rnn_targets = self.texts[:, 1:]

        # Embeddings tensor
        self.embeddings_tensor = tf.get_variable("embeddings",
                initializer=tf.constant(self.embeddings), trainable=True)

        # Cell
        cell_stack = [tf.nn.rnn_cell.LSTMCell(cell_size) for i in range(cell_height)]
        self.cell = tf.nn.rnn_cell.MultiRNNCell(cell_stack)

        embedded = tf.nn.embedding_lookup(self.embeddings_tensor, self.rnn_inputs)

        init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=embedded,
                sequence_length=self.text_lengths-1, # -1 since we don't predict 1st word
                initial_state=init_state,
                )

        # Project onto vocab
        hidden_layer = tf.keras.layers.Dense(self.vocab_size, name="vocab_proj")
        self.logits = hidden_layer.apply(rnn_outputs)

        # Softmax over vocab
        self.pred_probs = tf.nn.softmax(self.logits, axis=2) # Over vocab axis

        # Loss, take exp of to get perplexity
        sequence_mask = tf.sequence_mask(self.text_lengths-1, dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits, targets=self.rnn_targets,
                average_across_batch=False, weights=sequence_mask)

        self.total_loss = tf.reduce_mean(self.loss, axis=0)

        # Features
        self.full_perplexity = tf.exp(self.loss)

        context_mask = tf.sequence_mask((self.text_lengths-1 - self.end_lengths),
                dtype=tf.float32, maxlen=tf.reduce_max(self.text_lengths)-1
                ) # 1.0 for context
        ending_mask = sequence_mask - context_mask # 1.0 for endings

        self.ending_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits, targets=self.rnn_targets,
                average_across_batch=False, weights=ending_mask)
        self.end_perplexity = tf.exp(self.ending_loss)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        grads_and_vars = self.optimizer.compute_gradients(self.total_loss)
        gradients = [p[0] for p in grads_and_vars]
        variables = [p[1] for p in grads_and_vars]

        clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip_norm)
        apply_pairs = list(zip(clipped_gradients, variables))

        self.train_op = self.optimizer.apply_gradients(apply_pairs,
                global_step=self.global_step, name="apply_grads")

        # TB Summary
        self.summary = tf.summary.merge([
                tf.summary.scalar("training_loss", self.total_loss)
            ])

    def get_feed_dict(self, contexts, endings, **kwargs):
        batch_size = len(contexts)

        end_len = [len(ends[0]) for ends in endings]

        texts_concat = [sum(contexts[bi], [])+endings[bi][0] for bi in range(batch_size)]

        # Same steps for context and each ending
        sequence_matrices = []
        length_vectors = []

        lengths = np.array([len(t) for t in texts_concat])

        texts_matrix = np.zeros((batch_size, lengths.max()))
        for t_i in range(batch_size):
            texts_matrix[t_i, :lengths[t_i]] = texts_concat[t_i]

        return {
                self.texts: texts_matrix,
                self.text_lengths: lengths,
                self.end_lengths: end_len,
                self.batch_size: batch_size,
                }

    def train_step(self, context_batch, end_batch, sess, summary_writer, **kwargs):
        fetches = [
            self.total_loss,
            self.global_step,
            self.train_op,
            self.summary,
        ]

        feed_dict = self.get_feed_dict(context_batch, end_batch)

        output = sess.run(fetches, feed_dict)
        train_loss = output[0]
        step = output[1]

        # Write to summary
        summary_writer.add_summary(output[3], step)

        return train_loss, step

    def predict(self, contexts, endings, sess, **kwargs):
        e0 = [[e[0]] for e in endings]
        e1 = [[e[1]] for e in endings]

        fd0 = self.get_feed_dict(contexts, e0)
        fd1 = self.get_feed_dict(contexts, e1)

        fetches = [
            self.loss, # Fetches loss for sample in batch
        ]

        output0 = sess.run(fetches, fd0)
        output1 = sess.run(fetches, fd1)

        return np.argmin(np.stack([output0[0], output1[0]], axis=1), axis=1)

    def get_features(self):
        return [tf.expand_dims(self.full_perplexity, axis=1),
                tf.expand_dims(self.end_perplexity, axis=1)]
