import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")
from models.potr.data_process import train_preprocess

class POTRLoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        if self.args.loss_fn == 'mse':
            self.loss_fn = self.loss_mse
        elif self.args.loss_fn == 'smoothl1':
            self.loss_fn = self.smooth_l1
        elif self.args.loss_fn == 'l1':
            self.loss_fn = self.loss_l1
        else:
            raise ValueError('Unknown loss name {}.'.format(self.args.loss_fn))

    def smooth_l1(self, decoder_pred, decoder_gt):
        l1loss = nn.SmoothL1Loss(reduction='mean')
        return l1loss(decoder_pred, decoder_gt)

    def loss_l1(self, decoder_pred, decoder_gt, reduction='mean'):
        return nn.L1Loss(reduction=reduction)(decoder_pred, decoder_gt)

    def loss_activity(self, logits, class_gt):                                     
        """Computes entropy loss from logits between predictions and class."""
        return nn.functional.cross_entropy(logits, class_gt, reduction='mean')

    def compute_class_loss(self, class_logits, class_gt):
        """Computes the class loss for each of the decoder layers predictions or memory."""
        class_loss = 0.0
        for l in range(len(class_logits)):
            class_loss += self.loss_activity(class_logits[l], class_gt)

        return class_loss/len(class_logits)

    def layerwise_loss_fn(self, decoder_pred, decoder_gt, class_logits=None, class_gt=None):
        """Computes layerwise loss between predictions and ground truth."""
        pose_loss = 0.0

        for l in range(len(decoder_pred)):
            pose_loss += self.loss_fn(decoder_pred[l], decoder_gt)

        pose_loss = pose_loss/len(decoder_pred)
        
        class_loss = None
        if class_logits is not None:
            class_loss = self.compute_class_loss(class_logits, class_gt)
        

        return pose_loss, class_loss

    def ua_loss(self, decoder_pred, decoder_gt, class_logits, class_gt, uncertainty_matrix=None):
        B = decoder_gt.shape[0]
        T = decoder_gt.shape[-3]
        L = len(decoder_pred)

        pose_loss = 0.0
        class_loss = None
        uncertainty_loss = None

        loss_fn = nn.L1Loss(reduction='none')
        if uncertainty_matrix is not None:
            assert class_gt is not None
            assert uncertainty_matrix.shape == (self.args.num_activities, self.args.n_major_joints)
            uncertainty_vector = uncertainty_matrix[class_gt].reshape(B, 1, self.args.n_major_joints, 1).to(self.args.device) # (n_joints, )
            u_coeff = (torch.arange(1, T+1) / T).reshape(1, T, 1, 1).to(self.args.device)
        else:
            uncertainty_vector = 1
            u_coeff = 0

        for l in range(L):
            pose_loss += ((1 - u_coeff ** uncertainty_vector) * loss_fn(decoder_pred[l], decoder_gt)).mean()
        
        pose_loss = pose_loss / L

        
        if class_logits is not None:
            class_loss = self.compute_class_loss(class_logits, class_gt)     

        if uncertainty_matrix is not None:
            uncertainty_loss = torch.log(uncertainty_matrix).mean()


        return pose_loss, class_loss, uncertainty_loss

    def compute_loss(self, inputs=None, target=None, preds=None, class_logits=None, class_gt=None):
        return self.layerwise_loss_fn(preds, target, class_logits, class_gt)



    def forward(self, model_outputs, input_data):
        input_data = train_preprocess(input_data, self.args)
        
        '''selection_loss = 0
        if self.args.query_selection:
            prob_mat = model_outputs['mat'][-1]
            selection_loss = self.compute_selection_loss(
                inputs=prob_mat, 
                target=input_data['src_tgt_distance']
            )'''

        pred_class, gt_class = None, None
        if self.args.predict_activity:
            gt_class = input_data['action_ids']
            pred_class = model_outputs['out_class']

        uncertainty_matrix = None
        if self.args.consider_uncertainty:
            uncertainty_matrix = model_outputs['uncertainty_matrix']


        pose_loss, activity_loss, uncertainty_loss = self.ua_loss(
                decoder_pred=model_outputs['pred_pose'], 
                decoder_gt=input_data['decoder_outputs'], 
                class_logits=pred_class, 
                class_gt=gt_class, 
                uncertainty_matrix=uncertainty_matrix
                )
      
        pl = pose_loss.item()
        step_loss = pose_loss #+ selection_loss

        if self.args.predict_activity:
            step_loss += self.args.activity_weight*activity_loss

        if self.args.consider_uncertainty:
            step_loss -= self.args.uncertainty_weight*uncertainty_loss
         
        outputs = {
            'loss': step_loss, 
            #'selection_loss': selection_loss,
            'pose_loss': pl,
            }

        if self.args.predict_activity:
            outputs['activity_loss'] = activity_loss.item()

        if self.args.consider_uncertainty:
            outputs['uncertainty_loss'] = uncertainty_loss.item()

        return outputs
