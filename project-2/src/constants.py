# CONSTANTS

root_dir = ".."
data_dir = "data"

val_file = "validation_stories.csv"
train_file = "train_stories.csv"
test_file_labelled = "test-stories_lab.csv"
test_file_unlabelled = "test-stories_no_lab.csv"

embeddings_file = "mini.h5"
embed_dim = 300

model_dir = "saved_models"
tb_dir = "tb_summaries"
config_file = "config.json"

rand_seed = 42

# Fractions to use for train, validation, test split of original validation data
validation_frac = 0.2

preprocessed_file = "preprocessed.obj"
sentiment_file = "sentiment_preprocessed.obj"
doc2vec_file = "doc2vec_preprocessed.obj"
combined_file = "combined_preprocessed.obj"

vocab_size = 20000
unk_token = "<unk>"

prediction_file = "prediction_file.csv"
doc2vec_model_fn = "doc2vec.model"
