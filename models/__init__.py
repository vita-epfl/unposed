from .zero_vel import ZeroVel
from .pv_lstm import PVLSTM
from .disentangled import Disentangled
from .derpof import DeRPoF
from  models.history_repeats_itself.history_repeats_itself import HistoryRepeatsItself
from .sts_gcn.sts_gcn import STsGCN
from .msr_gcn.msrgcn import MSRGCN
from .potr.potr import POTR
from .st_trans.ST_Trans import ST_Trans
from .pgbig.pgbig import PGBIG

MODELS = {'zero_vel': ZeroVel,
          'pv_lstm': PVLSTM,
          'disentangled': Disentangled,
          'derpof': DeRPoF,
          'history_repeats_itself': HistoryRepeatsItself,
          'potr': POTR,
          'sts_gcn': STsGCN,
          'msr_gcn': MSRGCN,
          'st_trans': ST_Trans,
          'pgbig': PGBIG ,
          }
