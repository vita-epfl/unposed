from .adam import adam
from .sgd import sgd
from .adamw import adamw
from .sam import sam

OPTIMIZERS = {'adam': adam,
              'sgd': sgd,
              'adamw': adamw,
              'sam': sam}
