import numpy as np
import sys
import os

import torch
import torch.nn as nn

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")
import potr.utils as utils
from potr import INIT_FUNC

class DecoderLayer(nn.Module):

    def __init__(self, 
                model_dim, 
                num_heads, 
                dim_ffn, 
                dropout, 
                init_fn_name, 
                pre_normalization,
                use_query_embedding):
        super(DecoderLayer, self).__init__()
        init_fn = INIT_FUNC[init_fn_name]
        self.use_query_embedding = use_query_embedding

        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(model_dim, dim_ffn)
        self.linear2 = nn.Linear(dim_ffn, model_dim)
        self.relu = nn.ReLU()

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        utils.weight_init(self.linear1, init_fn=init_fn)
        utils.weight_init(self.linear2, init_fn=init_fn)

        self.forward_fn = self.forward_pre if pre_normalization else self.forward_post

    def forward(self, 
                target_seq, 
                memory,
                pos_encodings, 
                query_embedding=None,
                mask_look_ahead=None, 
                mask_target_padding=None):

        return self.forward_fn(
            target_seq,
            memory,
            pos_encodings,
            query_embedding=query_embedding,
            mask_look_ahead=mask_look_ahead,
            mask_target_padding=mask_target_padding
        )

    def handle_query_embedding(self, sequence, embedding):
        """Handle """
        if self.use_query_embedding:
            return sequence + embedding
        return sequence


    def forward_post(self, 
                target_seq, 
                memory,
                pos_encodings, 
                query_embedding=None,
                mask_look_ahead=None, 
                mask_target_padding=None):

        if self.use_query_embedding:
            q = k = v = target_seq + query_embedding
        else:
            q = k = v =  target_seq + pos_encodings

        self_attn, self_attn_weights = self.self_attn(
            query=q, key=k, value=v, 
            attn_mask=mask_look_ahead,
            key_padding_mask=mask_target_padding
        )
        self_attn = self.dropout1(self_attn)
        out_self_attn = self.norm1(self_attn + target_seq)

        attn, attn_weights = self.multihead_attn(
            query=self.handle_query_embedding(out_self_attn, query_embedding), 
            key=self.handle_query_embedding(memory, pos_encodings), 
            value=memory)
        attn = self.dropout2(attn)
        out_attn = self.norm2(attn + out_self_attn)

        ffn_output = self.linear1(out_attn)
        ffn_output = self.relu(ffn_output)
        ffn_output = self.dropout4(ffn_output)
        ffn_output = self.linear2(ffn_output)

        ffn_output = self.dropout3(ffn_output)
        outputs = self.norm3(ffn_output + out_attn)

        return outputs, self_attn_weights, attn_weights

    def forward_pre(self, 
                target_seq_, 
                memory,
                pos_encodings, 
                query_embedding=None,
                mask_look_ahead=None, 
                mask_target_padding=None):
        """Forward pass of the layer with pre normalization.

        Args:
        target_seq: [target_seq_length, batch_size, model_dim]
        memory: [source_seq_length, batch_size, model_dim]
        mask_look_ahead: []
        mask_target_padding:
        """
        target_seq = self.norm1(target_seq_)
        # 1) Compute self attention with current sequence of inferred tokens
        # query is the same as key for self attention
        if self.use_query_embedding:
            # in case of using only the query embedding follow DETR [2] which drops
            # values to zero and uses only the query embeddings
            q = k = target_seq + query_embedding
            v = target_seq
        else:
            q = k = v =  target_seq + pos_encodings

        self_attn, self_attn_weights = self.self_attn(
            query=q, key=k, value=v,
            attn_mask=mask_look_ahead, 
            key_padding_mask=mask_target_padding
        )
        self_attn = self.dropout1(self_attn)
        out_self_attn = self.norm2(self_attn + target_seq_)

        # 2) Attend the encoder's memory given the comptued self attention
        attn, attn_weights = self.multihead_attn(
            query=self.handle_query_embedding(out_self_attn, query_embedding), 
            key=self.handle_query_embedding(memory, pos_encodings),
            value=memory)
        attn = self.dropout2(attn)
        out_attn = self.norm3(attn + out_self_attn)

        # 3) Compute pointwise embeding by expanding and projecting + dropout
        ffn_output = self.linear1(out_attn)
        ffn_output = self.relu(ffn_output)
        ffn_output = self.dropout4(ffn_output)
        ffn_output = self.linear2(ffn_output)

        ffn_output = self.dropout3(ffn_output)

        return ffn_output, self_attn_weights, attn_weights


class TransformerDecoder(nn.Module):
    """Transformer decoder module."""
    def __init__(self,
                num_layers,
                model_dim,
                num_heads,
                dim_ffn,
                dropout,
                init_fn_name,
                pre_normalization,
                use_query_embedding):
        super(TransformerDecoder, self).__init__()
        self.use_query_embedding = use_query_embedding
        self.num_layers = num_layers

        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                dim_ffn=dim_ffn,
                dropout=dropout,
                init_fn_name=init_fn_name,
                pre_normalization=pre_normalization,
                use_query_embedding=use_query_embedding) 
            for _ in range(num_layers)
            ]
        )

    def forward(self, 
                target_seq,
                memory,
                pos_encodings,
                query_embedding=None,
                mask_target_padding=None,
                mask_look_ahead=None,
                get_attn_weights=False):
        """Computes forward pass of decoder.

        Args:
        target_seq: [target_sequence_length, batch_size, model_dim].
        memory: [source_sequence_length, batch_size, model_dim].
        pos_encodings: [target_seq_length, model_dim].
        mask_look_ahead: [target_seq_length, model_dim].

        Returns:
        A tensor with the decoded attention with shape [target_sequence_length,
        batch_size, model_dim].
        """
        seq_length = target_seq.size()[0]
        output_list = []
        attn_weights_list = [] if get_attn_weights else None
        outputs = torch.zeros_like(target_seq) if self.use_query_embedding else target_seq

        for l in range(self.num_layers):
            outputs, self_attn_weights, attn_weights = self.decoder_stack[l](
                outputs, memory,
                pos_encodings=pos_encodings,
                query_embedding=query_embedding,
                mask_target_padding=mask_target_padding,
                mask_look_ahead=mask_look_ahead
            )
            if get_attn_weights:
                attn_weights_list.append(attn_weights)
            output_list.append(outputs)

        return output_list, attn_weights_list

if __name__ == '__main__':
    thispath = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, thispath+"/../")
    import potr.utils as utils

    seq_length = 55
    batch_size = 8
    model_dim = 128
    tgt_seq = torch.FloatTensor(seq_length, batch_size, model_dim).fill_(1)
    memory = torch.FloatTensor(seq_length, batch_size, model_dim).uniform_(0, 1)

    mask_look_ahead = utils.create_look_ahead_mask(seq_length)
    mask_look_ahead = torch.from_numpy(mask_look_ahead)

    encodings = torch.FloatTensor(seq_length, 1, model_dim).uniform_(0,1)

    decoder = TransformerDecoder(
                                num_layers=25,
                                model_dim=128,
                                num_heads=2,
                                dim_ffn=16,
                                dropout=0.5,
                                init_fn_name='xavier',
                                pre_normalization=True,
                                use_query_embedding=False)
    outputs, attn_weights_list = decoder(tgt_seq, memory, encodings, mask_look_ahead=mask_look_ahead)

