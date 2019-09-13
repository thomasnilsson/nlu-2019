import data_loader as dl
import constants as C

from models.combinable_models import MODELS, get_model_class, get_model_info
from models.combiner import Combiner

MODELS["combiner"] = [Combiner, {"loader": dl.load_combined,
                                 "file_name": C.combined_file}]
