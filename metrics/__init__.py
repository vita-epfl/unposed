from .pose_metrics import ADE, FDE, local_ade, local_fde, VIM, VAM, MSE
from .pose_metrics import F1, F3, F7, F9, F9, F13, F17, F21 #new

POSE_METRICS = {'ADE': ADE,
                'FDE': FDE,
                'local_ade': local_ade,
                'local_fde': local_fde,
                'VIM': VIM,
                'VAM': VAM,
                'MSE': MSE,
                #new:
                'F1': F1,
                'F3': F3,
                'F7': F7,
                'F9': F9,
                'F13': F13,
                'F17': F17,
                'F21': F21
                }
