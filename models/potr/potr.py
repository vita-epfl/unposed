import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

from potr.transformer import Transformer
from potr.position_encodings import PositionEncodings1D
from potr import utils
from potr.pose_encoder_decoder import select_pose_encoder_decoder_fn
from models.potr.data_process import train_preprocess, post_process_to_euler

class POTR(nn.Module):
    def __init__(self, args):
        super(POTR, self).__init__()
        self.args = args

        pose_embedding_fn, pose_decoder_fn = \
            select_pose_encoder_decoder_fn(args)

        self.pose_embedding = pose_embedding_fn(args)#args.pose_embedding              
        self.pose_decoder = pose_decoder_fn(args) #args.pose_decoder   

        self.transformer = Transformer(args)

        transformer_params = filter(lambda p: p.requires_grad, self.transformer.parameters())
        n_transformer_params = sum([np.prod(p.size()) for p in transformer_params])
        thisname = self.__class__.__name__
        print('[INFO] ({}) Transformer has {} parameters!'.format(thisname, n_transformer_params))


        self.pos_encoder = PositionEncodings1D(
            num_pos_feats=args.model_dim,
            temperature=args.pos_enc_beta,
            alpha=args.pos_enc_alpha
        )

        self.pos_decoder = PositionEncodings1D(
            num_pos_feats=args.model_dim,
            temperature=args.pos_enc_beta,
            alpha=args.pos_enc_alpha
        )

        args.use_class_token = False
        self.init_position_encodings()
        self.init_query_embedding()

        if args.use_class_token:
            self.init_class_token()

        if args.predict_activity:
            self.action_head_size = self.model_dim if self.args.use_class_token \
                else args.model_dim*(args.obs_frames_num-1)
            self.action_head = nn.Sequential(
                nn.Linear(self.action_head_size, args.num_activities),
            )

        if args.consider_uncertainty:
            self.uncertainty_matrix = nn.parameter.Parameter(data=torch.zeros(args.num_activities, args.n_major_joints), requires_grad=True)
            nn.init.xavier_uniform_(self.uncertainty_matrix.data)
            #self.uncertainty_matrix.data


    def init_query_embedding(self):
        """Initialization of query sequence embedding."""
        self.query_embed = nn.Embedding(self.args.future_frames_num, self.args.model_dim)
        print('[INFO] ({}) Init query embedding!'.format(self.__class__.__name__))
        nn.init.xavier_uniform_(self.query_embed.weight.data)
        # self._query_embed.weight.data.normal_(0.0, 0.004)

    def init_class_token(self):
        token = torch.FloatTensor(1, self.args.model_dim)
        print('[INFO] ({}) Init class token!'.format(self.__class__.__name__))
        self.class_token = nn.Parameter(token, requires_grad=True)
        nn.init.xavier_uniform_(self.class_token.data)

    def init_position_encodings(self):
        src_len = self.args.obs_frames_num-1
        
        # when using a token we need an extra element in the sequence
        if self.args.use_class_token:
            src_len = src_len + 1
        
        encoder_pos_encodings = self.pos_encoder(src_len).view(
                src_len, 1, self.args.model_dim)

        decoder_pos_encodings = self.pos_decoder(self.args.future_frames_num).view(
                self.args.future_frames_num, 1, self.args.model_dim)

        mask_look_ahead = torch.from_numpy(
            utils.create_look_ahead_mask(
                self.args.future_frames_num, self.args.non_autoregressive))

        self.encoder_pos_encodings = nn.Parameter(
            encoder_pos_encodings, requires_grad=False)

        self.decoder_pos_encodings = nn.Parameter(
            decoder_pos_encodings, requires_grad=False)

        self.mask_look_ahead = nn.Parameter(
            mask_look_ahead, requires_grad=False)

    def handle_class_token(self, input_pose_seq):
        _, B, _ = input_pose_seq.size()
        token = self.class_token.squeeze().repeat(1, B, 1)
        input_pose_seq = torch.cat([token, input_pose_seq], axis=0)

        return input_pose_seq

    def handle_copy_query(self, indices, input_pose_seq_):

        batch_size = input_pose_seq_.size()[0]
        decoder_inputs = torch.FloatTensor(
            batch_size,
            self.args.future_frames_num,
            self.args.pose_dim * self.args.n_joints
        ).to(self.decoder_pos_encodings.device)
        for i in range(batch_size):
            for j in range(self.args.future_frames_num):
                src_idx, tgt_idx = indices[i][0][j], indices[i][1][j]
                decoder_inputs[i, tgt_idx] = input_pose_seq_[i, src_idx]
        dec_inputs_encode = self.pose_embedding(decoder_inputs)

        return torch.transpose(decoder_inputs, 0, 1), \
            torch.transpose(dec_inputs_encode, 0, 1)

    def forward(self, 
              inputs,
              mask_target_padding=None,
              get_attn_weights=False):

        preprocessed_inputs = train_preprocess(inputs, self.args)
        enc_shape = preprocessed_inputs['encoder_inputs'].shape
        dec_shape = preprocessed_inputs['decoder_inputs'].shape
        input_pose_seq = preprocessed_inputs['encoder_inputs'].reshape((*enc_shape[:-2], -1))
        target_pose_seq = preprocessed_inputs['decoder_inputs'].reshape((*dec_shape[:-2], -1))

        return self.forward_training(
                input_pose_seq, target_pose_seq, mask_target_padding, get_attn_weights)


    def forward_training(self,
                       input_pose_seq_,
                       target_pose_seq_,
                       mask_target_padding,
                       get_attn_weights):
                       
        input_pose_seq = input_pose_seq_
        target_pose_seq = target_pose_seq_
        if self.pose_embedding is not None:
            input_pose_seq = self.pose_embedding(input_pose_seq)
            target_pose_seq = self.pose_embedding(target_pose_seq)

        input_pose_seq = torch.transpose(input_pose_seq, 0, 1)
        target_pose_seq = torch.transpose(target_pose_seq, 0, 1)

        def query_copy_fn(indices):
            return self.handle_copy_query(indices, input_pose_seq_)

        if self.args.use_class_token:
            input_pose_seq = self.handle_class_token(input_pose_seq)

        
        attn_output, memory, attn_weights, enc_weights, mat = self.transformer(
            input_pose_seq,
            target_pose_seq,
            query_embedding=self.query_embed.weight,
            encoder_position_encodings=self.encoder_pos_encodings,
            decoder_position_encodings=self.decoder_pos_encodings,
            mask_look_ahead=self.mask_look_ahead,
            mask_target_padding=mask_target_padding,
            get_attn_weights=get_attn_weights,
            query_selection_fn=query_copy_fn
        )

        end = self.args.pose_dim * self.args.n_major_joints
        out_sequence = []
        target_pose_seq_ = mat[0] if self.args.query_selection else \
            torch.transpose(target_pose_seq_, 0, 1)

        for l in range(self.args.num_decoder_layers):
            out_sequence_ = self.pose_decoder(
                attn_output[l].view(-1, self.args.model_dim))
            out_sequence_ = out_sequence_.view(
                self.args.future_frames_num, -1, self.args.pose_dim * self.args.n_major_joints)
            out_sequence_ = out_sequence_ + target_pose_seq_[:, :, 0:end]
            
            out_sequence_ = torch.transpose(out_sequence_, 0, 1)

            shape = out_sequence_.shape
            out_sequence_ = out_sequence_.reshape((*shape[:-1], self.args.n_major_joints, self.args.pose_dim))
            out_sequence.append(out_sequence_)


        pred_euler_pose = torch.tensor(post_process_to_euler( # convert to post_process_to_format
            out_sequence[-1].detach().cpu().numpy(), 
            self.args.n_major_joints, 
            self.args.n_h36m_joints, 
            self.args.pose_format))

        outputs = {
                'pred_metric_pose': pred_euler_pose,
                'pred_pose': out_sequence,
                'attn_weights': attn_weights,
                'enc_weights': enc_weights,
                'mat': mat
        }

        if self.args.predict_activity:
            outputs['out_class'] = self.predict_activity(attn_output, memory)

        if self.args.consider_uncertainty:
            outputs['uncertainty_matrix'] = torch.sigmoid(self.uncertainty_matrix)


        return outputs



    def predict_activity(self, attn_output, memory):
        in_act = torch.transpose(memory, 0, 1)

        if self.args.use_class_token:
            token = in_act[:, 0]
            actions = self.action_head(token)
            return [actions]      

        in_act = torch.reshape(in_act, (-1, self.action_head_size))
        actions = self.action_head(in_act)
        return [actions]


if __name__ == '__main__':
    thispath = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, thispath+"/../")
    import potr.utils as utils

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_encoder_layers', default=4)
    parser.add_argument('--num_decoder_layers', default=4)
    parser.add_argument('--query_selection', default=False)
    parser.add_argument('--future_frames_num', default=20)
    parser.add_argument('--obs_frames_num', default=50)
    parser.add_argument('--use_query_embedding', default=False)
    parser.add_argument('--num_layers', default=6)
    parser.add_argument('--model_dim', default=128)
    parser.add_argument('--num_heads', default=2)
    parser.add_argument('--dim_ffn', default=16)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--init_fn_name', default='xavier')
    parser.add_argument('--pre_normalization', default=True)
    parser.add_argument('--pose_dim', default=9)
    parser.add_argument('--pose_embedding_type', default='gcn_enc')
    parser.add_argument('--pos_enc_beta', default=500)
    parser.add_argument('--pos_enc_alpha', default=10)
    parser.add_argument('--use_class_token', default=False)
    parser.add_argument('--predict_activity', default=True)
    parser.add_argument('--num_activities', default=15)
    parser.add_argument('--non_autoregressive', default=True)
    parser.add_argument('--n_joints', default=21)
    parser.add_argument('--pose_format', default='rotmat')
    #parser.add_argument('--pose_embedding')
    #parser.add_argument('--')
    #parser.add_argument('--')
    args = parser.parse_args()


    src_seq_length = args.obs_frames_num
    tgt_seq_length = args.future_frames_num
    batch_size = 8

    src_seq = torch.FloatTensor(batch_size, src_seq_length - 1, args.pose_dim*args.n_joints).uniform_(0, 1)
    tgt_seq = torch.FloatTensor(batch_size, tgt_seq_length, args.pose_dim*args.n_joints).fill_(1)
    

    #mask_look_ahead = utils.create_look_ahead_mask(tgt_seq_length)
    #mask_look_ahead = torch.from_numpy(mask_look_ahead)

    #encodings = torch.FloatTensor(tgt_seq_length, 1, args.model_dim).uniform_(0,1)

    model = POTR(args)
    
    out_attn, memory, out_weights_, enc_weights_, (tgt_plain, prob_matrix_) = model(src_seq,
                       tgt_seq,
                       None,
                       get_attn_weights=False)