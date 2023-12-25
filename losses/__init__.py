from .mse_vel import MSEVel
from .mse_pose import MSEPose
from .mae_vel import MAEVel
from .derpof_loss import DeRPoFLoss
from .his_rep_itself_loss import HisRepItselfLoss
from .mpjpe import MPJPE
from .msr_gcn_loss import MSRGCNLoss
from .potr_loss import POTRLoss
from .pua_loss import PUALoss
from .pgbig_loss import PGBIG_PUALoss

LOSSES = {'mse_vel': MSEVel,
          'mse_pose': MSEPose,
          'mae_vel': MAEVel,
          'derpof': DeRPoFLoss,
          'his_rep_itself': HisRepItselfLoss,
          'mpjpe': MPJPE,
          'msr_gcn':MSRGCNLoss,
          'potr': POTRLoss,
          'pua_loss': PUALoss,
          'pgbig_loss': PGBIG_PUALoss
          }
