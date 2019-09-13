import json
import os
import numpy as np
import tensorflow as tf

import constants as C

from models.model import Model
import models.combinable_models as ml


class Combiner(Model):
    def __init__(self, load_dirs, sess, learning_rate=1e-3,
                 hidden_size=None, fine_tune=False, **kwargs):
        super(Combiner, self).__init__(**kwargs)

        # Only train combination on multi-ending data
        self.train_single = False
        self.train_multi = True

        # Handle own initialization
        self.initialize = False

        model_list = [
            {"name": "rnn", "load": True},
            {"name": "sentiment", "load": True},
            {"name": "lm", "load": True},
            {"name": "features", "load": False},
            {"name": "doc2vec_combiner", "load": True},
        ]

        # Add in load directories
        dir_i = 0
        for mod_dict in model_list:
            if mod_dict["load"]:
                assert dir_i < len(load_dirs), "Wrong amount of load directories given"

                mod_dict["dir"] = load_dirs[dir_i]
                dir_i += 1

            mod_dict["class"] = ml.get_model_class(mod_dict["name"])
            mod_dict["load_config"] = ml.get_model_info(mod_dict["name"])["file_name"]
        assert dir_i == len(load_dirs), "Wrong amount of load directories given"

        # Instantiate models
        self.models = []
        self.data_configs = {}
        for model_dict in model_list:
            with tf.variable_scope(model_dict["name"]):
                # Always feed these
                if model_dict["name"] != "doc2vec_combiner":
                    model_config = {
                        "embeddings": kwargs["embeddings"][model_dict["load_config"]],
                        "word_to_id": kwargs["word_to_id"][model_dict["load_config"]],
                        "id_to_word": kwargs["id_to_word"][model_dict["load_config"]],
                    }

                if model_dict["load"]:
                    if model_dict["name"] == "doc2vec_combiner":
                        model_config["path_to_d2v"] = os.path.join(model_dict["dir"], C.doc2vec_model_fn)
                    else:
                        config_file_path = os.path.join(model_dict["dir"], C.config_file)
                        loaded_config = json.load(open(config_file_path))
                        model_config.update(loaded_config)

                new_model = model_dict["class"](**model_config)

                # Add to correct data_config list
                if model_dict["load_config"] in self.data_configs:
                    self.data_configs[model_dict["load_config"]].append(new_model)
                else:
                    self.data_configs[model_dict["load_config"]] = [new_model]

                self.models.append(new_model)

        # Construct combination
        with tf.variable_scope("combination"):
            self.global_step = tf.train.get_or_create_global_step()

            self.labels = tf.placeholder(dtype=tf.float32, shape=(None,), name="labels")

            features = sum([m.get_features() for m in self.models], [])
            self.tf_features = tf.concat(features, axis=1, name="combined_features")

            if hidden_size:
                hidden_layer = tf.keras.layers.Dense(hidden_size,
                                                     activation=tf.nn.sigmoid)
                hidden_output = hidden_layer.apply(self.tf_features)
            else:
                hidden_output = self.tf_features

            final_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
            self.logit = final_layer.apply(self.tf_features)[:, 0]

        # Loss
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.logit))

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        if fine_tune:
            training_vars = tf.trainable_variables()
        else:
            training_vars = tf.trainable_variables(scope="combination")

        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=self.global_step,
                                                var_list=training_vars)

        # Initialize
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        for model_dict in model_list:
            if model_dict["load"] and model_dict["name"] != "doc2vec_combiner":  # Nothing to restore for doc2vec
                cp_file = tf.train.latest_checkpoint(model_dict["dir"])

                model_vars = tf.trainable_variables(scope=model_dict["name"])
                var_names = [v.name.split(":")[0] for v in model_vars]
                original_var_names = [n.split("/", 1)[-1] for n in var_names]
                var_map = dict(zip(original_var_names, model_vars))

                # Make a saver that only loads variables for one model
                saver = tf.train.Saver(var_list=var_map)
                saver.restore(sess, cp_file)

    def get_feed_dict(self, contexts, endings, end_i=0):
        feed_dict = {}

        for cfg_name in self.data_configs:
            cfg_contexts = [c[cfg_name] for c in contexts]
            # Note, there are 2 endings
            cfg_endings = [[e[cfg_name][end_i]] for e in endings]

            for model in self.data_configs[cfg_name]:
                model_fd = model.get_feed_dict(cfg_contexts, cfg_endings)
                feed_dict.update(model_fd)

        return feed_dict

    def train_step(self, contexts, endings, sess, labels=None, **kwargs):
        fetches = [
            self.loss,
            self.global_step,
            self.train_op,
        ]
        feed_dicts = [self.get_feed_dict(contexts, endings, end_i=i) for i in [0, 1]]

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
            self.logit,
        ]

        feed_dicts = [self.get_feed_dict(contexts, endings, end_i=i) for i in [0, 1]]

        outputs = []
        for fd in feed_dicts:

            output = sess.run(fetches, fd)
            outputs.append(output)

        both_logits = np.stack([o[0] for o in outputs], axis=1)
        return np.argmax(both_logits, axis=1)
