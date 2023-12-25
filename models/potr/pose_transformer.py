import numpy as np
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import models.potr.utils.utils as utils
import models.potr.utils.position_encodings as PositionEncodings
import models.potr.transformer_encoder as Encoder
import models.potr.transformer_decoder as Decoder
from models.potr.transformer import Transformer
import models.potr.pose_encoder_decoder as PoseEncoderDecoder

class PoseTransformer(nn.Module):
    def __init__(self, args):
        super(PoseTransformer, self).__init__()
        """
        args:
            training: (bool) if the it's training mode, true, else, false
            non_autoregressive: (boo

            transformer arguments:
                num_encoder_layers: (int)
                num_decoder_layers: (int)
                model_dim: (int)
                num_heads: (int)
                dim_ffn: (int)
                dropout: (float)
                init_fn: (string)
                use_query_embedding: (bool)
                pre_normalization: (bool)
                query_selection: (bool)
                pred_frames_num: (int)

            position_encoder and position_decoder arguments:
                pose_enc_beta: (float)
                pose_enc_alpha: (float)
                # pos_encoding_params = (args.pose_enc_beta, args.pos_enc_alpha)

            handle_copy_query:
                pose_dim: (int)

            init_position_encodings:
                obs_frames_num: (int)

            PoseEncoderDecoder.select_pose_encoder_decoder:
                pose_embedding_type: (str)
                pose_format: (str)
                n_joints: (int)

            forward_training:
                input_dim: (int)
                
            useless:
                use_class_token: False



        """

        self.args = args
        
        """
        self.init_fn = utils.normal_init_ \
            if args.init_fn == 'normal_init' else utils.xavier_init_ #utils.xavier_init_
        """
        pose_embedding_fn, pose_decoder_fn = \
            PoseEncoderDecoder.select_pose_encoder_decoder_fn(args.pose_embedding_type)                                                                   
        self.pose_embedding = pose_embedding_fn(args)#args.pose_embedding              
        self.pose_decoder = pose_decoder_fn(args) #args.pose_decoder   


        self.transformer = Transformer(
            num_encoder_layers=self.args.num_encoder_layers,
            num_decoder_layers=self.args.num_decoder_layers,
            model_dim=self.args.model_dim,
            num_heads=self.args.num_heads,
            dim_ffn=self.args.dim_ffn,
            dropout=self.args.dropout,
            init_fn=self.args.init_fn,
            use_query_embedding=self.args.use_query_embedding,
            pre_normalization=self.args.pre_normalization,
            query_selection=self.args.query_selection,
            pred_frames_num=self.args.pred_frames_num           
        )

        self.position_encoder = PositionEncodings.PositionEncodings1D(
            num_position_feats=self.args.model_dim,
            temperature=self.args.pose_enc_beta,
            alpha=self.args.pose_enc_alpha
        )

        self.position_decoder = PositionEncodings.PositionEncodings1D(
            num_position_feats=self.args.model_dim,
            temperature=self.args.pose_enc_beta,
            alpha=self.args.pose_enc_alpha
        )

        self.init_position_encodings() # self.query_embedding
        self.init_query_embedding() 
        # self.encoder_pose_encodings, 
        # self.decoder_pose_encodings, 
        # self.mask_look_ahead

    def init_query_embedding(self):
        """Initialization of query sequence embedding."""
        self.query_embedding = nn.Embedding(self.args.pred_frames_num, self.args.model_dim)
        print('[INFO] ({}) Init query embedding!'.format(self.__class__.__name__))
        nn.init.xavier_uniform_(self.query_embedding.weight.data)
        # self._query_embed.weight.data.normal_(0.0, 0.004)       

    def init_position_encodings(self):
        src_len = self.args.obs_frames_num - 1
        # when using a token we need an extra element in the sequence
        
        encoder_position_encodings = self.position_encoder(src_len).view(
                src_len, 1, self.args.model_dim)
        decoder_position_encodings = self.position_decoder(self.args.pred_frames_num).view(
                self.args.pred_frames_num, 1, self.args.model_dim)
        mask_look_ahead = torch.from_numpy(
            utils.create_look_ahead_mask(
                self.args.pred_frames_num, self.args.non_autoregressive))
        
        self.encoder_position_encodings = nn.Parameter(
            encoder_position_encodings, requires_grad=False)
        self.decoder_position_encodings = nn.Parameter(
            decoder_position_encodings, requires_grad=False)
        self.mask_look_ahead = nn.Parameter(
            mask_look_ahead, requires_grad=False)

    def forward(self, inputs):
        if self.args.training:
            return self.forward_training(inputs)
        elif self.args.non_autoregressive:
            return self.forward_training(inputs)
        else:
            return self.forward_autoregressive(inputs)

    def handle_copy_query(self, indices, observed_expmap_pose):
        """Handles the way queries are generated copying items from the inputs.

        Args:
        indices: A list of tuples len `batch_size`. Each tuple contains has the
            form (input_list, target_list) where input_list contains indices of
            elements in the input to be copy to elements in the target specified by
            target_list.
        input_pose_seq_: Source skeleton sequence [batch_size, src_len, pose_dim].

        Returns:
            A tuple with first elements the decoder input skeletons with shape
            [tgt_len, batch_size, skeleton_dim], and the skeleton embeddings of the 
            input sequence with shape [tgt_len, batch_size, pose_dim].
        """
        batch_size = observed_expmap_pose.size()[0]
        decoder_inputs = torch.FloatTensor(
            batch_size,
            self.args.pred_frames_num,
            self.args.pose_dim
        ).to(self.decoder_position_encodings.device)
        for i in range(batch_size):
            for j in range(self._target_seq_length):
                src_idx, tgt_idx = indices[i][0][j], indices[i][1][j]
                decoder_inputs[i, tgt_idx] = observed_expmap_pose[i, src_idx]
            dec_inputs_encode = self.pose_embedding(decoder_inputs)

        return torch.transpose(decoder_inputs, 0, 1), \
            torch.transpose(dec_inputs_encode, 0, 1)   

    def forward_training(self, inputs):
        
        observed_expmap_pose = inputs['observed_expmap_pose']
        future_expmap_pose = inputs['future_expmap_pose']
        
        # 1. Encode the sequence with given pose encoder
        # [batch_size, sequence_length, model_dim]
        if self.pose_embedding is not None:
            observed_expmap_pose = self.pose_embedding(observed_expmap_pose)
            future_expmap_pose = self.pose_embedding(future_expmap_pose)
        
        # 2. Compute the look-ahead mask and the positional encodings
        # [sequence_length, batch_size, model_dim]
        observed_expmap_pose = torch.transpose(observed_expmap_pose, 0, 1)
        future_expmap_pose = torch.transpose(future_expmap_pose, 0, 1)

        def query_copy_fn(indices):
            return self.handle_copy_query(indices, observed_expmap_pose)       
        # 3. Compute the attention weights using the transformer
        # [future_expmap_pose_length, batch_size, model_dim]

        attn_output, memory, attn_weights, enc_weights, mat = self.transformer(
            observed_expmap_pose,
            future_expmap_pose,
            query_embedding=self.query_embedding.weight,
            encoder_position_encoding=self.encoder_position_encodings,
            decoder_position_encoding=self.decoder_position_encodings,
            mask_look_ahead=self.mask_look_ahead,
            mask_target_padding=inputs['mask_target_padding'],
            get_attn_wights=inputs['get_attn_weights'],
            query_selection=query_copy_fn
        )

        end = self.args.input_dim if self.args.input_dim == self.args.pose_dim else self.args.pose_dim
        out_sequence = []
        future_expmap_pose = mat[0] if self.args.query_selection else \
            torch.transpose(inputs['future_expmap_pose'], 0, 1)

        # 4. Decode qequence with pose decoder.
        # The decoding process is time independent.
        # It means non-autoregressive ar parallel decoding.
        # [batch_size, pred_frames_num, pose_dim]
        for l in range(self.args.num_decoder_layers):
            # [pred_frames_num*batch_size, pose_dim]
            out_sequence_ = self.pose_decoder(
                attn_output[l].view(-1, self.args.model_dim))
            # [pred_frames_num, batch_size, pose_dim]
            out_sequence_ = out_sequence_.view(
                self.args.pred_frames_num, -1, self.args.pose_dim)
            # apply residual connection between target query and predicted pose
            # [pred_frames_num, batch_size, pose_dim]
            out_sequence_ = out_sequence_ + future_expmap_pose[:, :, 0:end]
            # [batch_size, pred_frames_num, pose_dim]
            out_sequence_ = torch.transpose(out_sequence_, 0, 1)
            out_sequence.append(out_sequence_)

        
        outputs = {
            'pred_expmap_pose': out_sequence,
            'attn_weights': attn_weights,
            'enc_weights': enc_weights,
            'mat': mat
        }

        return outputs

    def forward_autoregressive(self, inputs):
        pass
