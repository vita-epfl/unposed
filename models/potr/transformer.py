import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
import os
import sys
thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

from potr.transformer_encoder import TransformerEncoder
from potr.transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, transformer_args):
        super(Transformer, self).__init__()
        self.args = transformer_args
        self.use_query_embedding = self.args.use_query_embedding
        self.query_selection = self.args.query_selection

        self.encoder = TransformerEncoder(
            num_layers=self.args.num_encoder_layers,
            model_dim=self.args.model_dim,
            num_heads=self.args.num_heads,
            dim_ffn=self.args.dim_ffn,
            dropout=self.args.dropout,
            init_fn_name=self.args.init_fn_name,
            pre_normalization=self.args.pre_normalization
        )

        self.decoder = TransformerDecoder(
            num_layers=self.args.num_decoder_layers,
            model_dim=self.args.model_dim,
            num_heads=self.args.num_heads,
            dim_ffn=self.args.dim_ffn,
            dropout=self.args.dropout,
            init_fn_name=self.args.init_fn_name,
            use_query_embedding=self.args.use_query_embedding,
            pre_normalization=self.args.pre_normalization
        )  

        if self.args.query_selection:
            self.position_predictor = nn.Linear(self.args.model_dim, self.args.future_frames_num)

    def process_index_selection(self, self_attn, one_to_one_selection=False):
        """Selection of query elments using position predictor from encoder memory.

        After prediction a maximum assignement problem is solved to get indices for
        each element in the query sequence.

        Args:
        self_attn: Encoder memory with shape [src_len, batch_size, model_dim]

        Returns:
        A tuple with two list of i and j matrix entries of m
        """
        batch_size = self_attn.size()[1]
        in_pos = torch.transpose(self_attn, 0, 1)
        prob_matrix = self.position_predictor(in_pos)

        if one_to_one_selection:
            soft_matrix = F.softmax(prob_matrix, dim=2)
            indices = [linear_sum_assignment(soft_matrix[i].cpu().detach(), maximize=True) 
                        for i in range(batch_size)]
        else:
            soft_matrix = F.softmax(prob_matrix, dim=1)
            indices_rows = torch.argmax(soft_matrix, 1)
            indices = [(indices_rows[i], list(range(prob_matrix.size()[2]))) 
                for i in range(batch_size)]

        return indices, soft_matrix 

    def forward(self,
                source_seq,
                target_seq,
                encoder_position_encodings=None,
                decoder_position_encodings=None,
                query_embedding=None,
                mask_target_padding=None,
                mask_look_ahead=None,
                get_attn_weights=False,
                query_selection_fn=None):

        if self.use_query_embedding:
            bs = source_seq.size()[1]
            query_embedding = query_embedding.unsqueeze(1).repeat(1, bs, 1)
            decoder_position_encodings = encoder_position_encodings
        
        memory, enc_weights = self.encoder(source_seq, encoder_position_encodings)

        tgt_plain = None
        if self.query_selection:
            indices, prob_matrix = self.process_index_selection(memory)
            tgt_plain, target_seq = query_selection_fn(indices)
        
        out_attn, out_weights = self.decoder(
            target_seq,
            memory,
            decoder_position_encodings,
            query_embedding=query_embedding,
            mask_target_padding=mask_target_padding,
            mask_look_ahead=mask_look_ahead,
            get_attn_weights=get_attn_weights
        )

        out_weights_ = None
        enc_weights_ = None
        prob_matrix_ = None
        if get_attn_weights:
            out_weights_, enc_weights_ = out_weights, enc_weights

        if self.query_selection:
            prob_matrix_ =  prob_matrix

        return out_attn, memory, out_weights_, enc_weights_, (tgt_plain, prob_matrix_)