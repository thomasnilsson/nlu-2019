import tensorflow as tf
import numpy as np

from models.model import Model
import constants as C

class RNNClassifier(Model):
    def __init__(self, learning_rate=1e-3, grad_clip_norm=10,
            cell_size=128, hidden_size=128, dropout=0, **kwargs):
        super(RNNClassifier, self).__init__(**kwargs)

        self.train_multi = True # Train on multi ending data

        # Save dropout prob. in var. to allow p=0 during prediction
        self.dropout_p = tf.placeholder_with_default(dropout, shape=())

        # Global step
        self.global_step = tf.train.get_or_create_global_step()

        # Placeholders
        self.context = tf.placeholder(shape=(None,None), dtype=tf.int32, name="context")
        self.end = tf.placeholder(shape=(None, None), dtype=tf.int32, name="end")
        self.context_length = tf.placeholder(dtype=tf.int32, name="context_length")
        self.end_length = tf.placeholder(dtype=tf.int32, name="end_length")
        self.labels = tf.placeholder(dtype=tf.float32, name="true_labels")

        # Embeddings tensor
        self.embeddings = tf.get_variable("embeddings",
                initializer=tf.constant(self.embeddings), trainable=True)

        # Cells
        self.fw_cell = tf.nn.rnn_cell.GRUCell(cell_size)
        self.bw_cell = tf.nn.rnn_cell.GRUCell(cell_size)

        # Attention variables
        self.attention_v = tf.get_variable(name="attention_v", shape=(cell_size),
                dtype=tf.float32)
        self.attention_w = tf.get_variable(name="attention_w",
                shape=(cell_size, cell_size),dtype=tf.float32)
        self.attention_b = tf.get_variable(name="attention_b",
                shape=(cell_size),dtype=tf.float32)

        # Concatenate fw and bw output states, (only use c-state)
        self.context_features = self.extract_features(self.context, self.context_length)
        self.end_features = self.extract_features(self.end, self.end_length)

        bilstm_features = tf.concat([self.context_features, self.end_features], axis=1)

        if dropout > 0:
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_p)
            self.features = dropout_layer.apply(bilstm_features)
            #self.features = tf.nn.dropout(bilstm_features, rate=self.dropout_p)
        else:
            self.features = bilstm_features

        # Dense layers
        if hidden_size:
            self.hidden_output = tf.layers.dense(self.features, hidden_size,
                    activation=tf.nn.relu, use_bias=True, name="rnn_hidden_layer")
        else:
            self.hidden_output = self.features

        self.logit = tf.layers.dense(self.hidden_output, 1, activation=None,
                use_bias=True, name="rnn_logit_layer")[:,0] # Make shape fit with labels

        # Loss
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.labels, logits=self.logit))

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        gradients = [p[0] for p in grads_and_vars]
        variables = [p[1] for p in grads_and_vars]

        clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip_norm)
        apply_pairs = list(zip(clipped_gradients, variables))

        self.train_op = self.optimizer.apply_gradients(apply_pairs,
                global_step=self.global_step, name="apply_grads")

    def extract_features(self, seq, seq_lengths):
        # Lookup word mebeddings
        embedded = tf.nn.embedding_lookup(self.embeddings, seq)

        # Run batch of sentences through rnn
        outputs, states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell,
                embedded, seq_lengths, dtype=tf.float32)

        fw_states = outputs[0] # Shape (batch_size, max_len, cell_size)
        bw_states = outputs[1]

        # Attention mechanism
        fw_linsum = tf.einsum("bsc,cd->bsd", fw_states, self.attention_w) +\
            self.attention_b
        bw_linsum = tf.einsum("bsc,cd->bsd", bw_states, self.attention_w) +\
            self.attention_b
        fw_tanh = tf.tanh(fw_linsum)
        bw_tanh = tf.tanh(bw_linsum)

        fw_scores = tf.einsum("bsd,d->bs", fw_tanh, self.attention_v)
        bw_scores = tf.einsum("bsd,d->bs", bw_tanh, self.attention_v)

        attention_weights_fw = tf.nn.softmax(fw_scores, axis=1)
        attention_weights_bw = tf.nn.softmax(bw_scores, axis=1)
        # Shape (batch_size, max_len)

        # Sum out max_len dimension
        fw_composed = tf.reduce_sum(fw_states *\
                tf.expand_dims(attention_weights_fw, axis=2), axis=1)
        bw_composed = tf.reduce_sum(bw_states *\
                tf.expand_dims(attention_weights_bw, axis=2), axis=1)

        # Concatenate composed states
        return tf.concat([fw_composed, bw_composed], axis=1)

    def prepare_feed_dicts(self, contexts, endings):
        batch_size = len(contexts)

        contexts_concat = [sum(c, []) for c in contexts]
        seqs = [contexts_concat]

        n_ends = len(endings[0])

        for ei in range(n_ends):
            seqs.append([e[ei] for e in endings])

        # Same steps for context and each ending
        sequence_matrices = []
        length_vectors = []

        for s in seqs:
            lengths = np.array([len(c) for c in s])
            length_vectors.append(lengths)

            s_matrix = np.zeros((batch_size, lengths.max()))
            for s_i in range(batch_size):
                s_matrix[s_i, :lengths[s_i]] = s[s_i]

            sequence_matrices.append(s_matrix)

        feed_dicts = []
        for ei in range(1,1+n_ends):
            feed_dicts.append({
                self.context: sequence_matrices[0],
                self.end: sequence_matrices[ei],
                self.context_length: length_vectors[0],
                self.end_length: length_vectors[ei],
            })

        return feed_dicts

    def train_step(self, context_batch, end_batch, sess, single=False,
            labels=None, **kwargs):
        fetches = [
            self.loss,
            self.global_step,
            self.train_op,
        ]

        feed_dicts = self.prepare_feed_dicts(context_batch, end_batch)

        # Add on labels
        feed_dicts[0][self.labels] = 1-labels
        feed_dicts[1][self.labels] = labels

        outputs = []
        for fd in feed_dicts:
            output = sess.run(fetches, fd)
            outputs.append(output)

        train_loss = np.mean([o[0] for o in outputs])
        step = outputs[-1][1]

        return train_loss, step

    def predict(self, contexts, endings, sess, **kwargs):
        fetches = [
            self.logit
        ]
        f0,f1 = self.prepare_feed_dicts(contexts, endings)

        # Deactivate dropout
        f0[self.dropout_p] = 0.0
        f1[self.dropout_p] = 0.0

        output0 = sess.run(fetches, f0)
        output1 = sess.run(fetches, f1)

        return np.argmax(np.stack([output0[0], output1[0]], axis=1), axis=1)

    def get_feed_dict(self, contexts, endings, **kwargs):
        fd = self.prepare_feed_dicts(contexts, endings)[0]

        # Deactivate dropout
        fd[self.dropout_p] = 0.0

        return fd

    def get_features(self):
        return [self.hidden_output]
