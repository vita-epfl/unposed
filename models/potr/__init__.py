import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")
from potr.utils import xavier_init, normal_init


INIT_FUNC = {
    'xavier': xavier_init,
    'normal': normal_init
}