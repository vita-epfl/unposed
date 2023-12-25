from argparse import ArgumentParser
import torch
import torch.nn as nn
import os 
import sys


thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")
from potr.pose_encoder_decoder import pose_decoder_mlp
from potr import utils
from potr import INIT_FUNC

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, dim_ffn, init_fn, pre_normalization):
        super(EncoderLayer, self).__init__()       

        init_fn = INIT_FUNC[init_fn]
        self.pre_normalization = pre_normalization

        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        self.linear1 = nn.Linear(model_dim, dim_ffn)
        self.linear2 = nn.Linear(dim_ffn, model_dim)
        self.norm1 = nn.LayerNorm(model_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(model_dim, eps=1e-5)

        utils.weight_init(self.linear1, init_fn=init_fn)
        utils.weight_init(self.linear2, init_fn=init_fn)

    def forward(self, source_seq, pos_encodings):
        """Computes forward pass according.

        Args:
        source_seq: [sequence_length, batch_size, model_dim].
        pos_encodings: [sequence_length, model_dim].

        Returns:
        Tensor of shape [sequence_length, batch_size, model_dim].
        """
        if self.pre_normalization:
            return self.forward_pre(source_seq, pos_encodings)

        return self.forward_post(source_seq, pos_encodings)

    def forward_post(self, source_seq, pos_encodings):
        """Computes decoder layer forward pass with pre normalization.

        Args:
        source_seq: [sequence_length, batch_size, model_dim].
        pos_encodings: [sequence_length, model_dim].

        Returns:
        Tensor of shape [sequence_length, batch_size, model_dim].
        """
        # add positional encodings to the input sequence
        # for self attention query is the same as key
        query = source_seq + pos_encodings
        key = query
        value = source_seq

        attn_output, attn_weights = self.self_attn(
            query, 
            key, 
            value, 
            need_weights=True
        )

        norm_attn = self.dropout_layer(attn_output) + source_seq
        norm_attn = self.norm1(norm_attn)

        output = self.linear1(norm_attn)
        output = self.relu(output)
        output = self.dropout_layer(output)
        output = self.linear2(output)
        output = self.dropout_layer(output) + norm_attn
        output = self.norm2(output)

        return output, attn_weights

    def forward_pre(self, source_seq_, pos_encodings):
        """Computes decoder layer forward pass with pre normalization.

        Args:
        source_seq: [sequence_length, batch_size, model_dim].
        pos_encodings: [sequence_length, model_dim].

        Returns:
        Tensor of shape [sequence_length, batch_size, model_dim].
        """
        # add positional encodings to the input sequence
        # for self attention query is the same as key
        source_seq = self.norm1(source_seq_)
        query = source_seq + pos_encodings
        key = query
        value = source_seq

        attn_output, attn_weights = self.self_attn(
            query, 
            key, 
            value, 
            need_weights=True
        )
        
        norm_attn_ = self.dropout_layer(attn_output) + source_seq_
        norm_attn = self.norm2(norm_attn_)

        output = self.linear1(norm_attn)
        output = self.relu(output)
        output = self.dropout_layer(output)
        output = self.linear2(output)
        output = self.dropout_layer(output) + norm_attn_

        return output, attn_weights




class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, dim_ffn, dropout, init_fn_name, pre_normalization):
        super(TransformerEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dim_ffn = dim_ffn
        self.dropout = dropout
        self.init_fn = INIT_FUNC[init_fn_name]
        self.pre_normalization = pre_normalization


        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                dim_ffn=self.dim_ffn,
                dropout=self.dropout,
                init_fn=init_fn_name, # (str)
                pre_normalization=self.pre_normalization)
                for s in range(self.num_layers)]
            )

    def forward(self, input_sequence, pos_encodings):
        """Computes decoder forward pass.

        Args:
        source_seq: [sequence_length, batch_size, model_dim].
        pos_encodings: [sequence_length, model_dim].

        Returns:
        Tensor of shape [sequence_length, batch_size, model_dim].
        """
        outputs = input_sequence
        for l in range(self.num_layers):
            outputs, attn_weights = self.encoder_stack[l](outputs, pos_encodings)

    #    if self._norm:
    #      outputs = self._norm(outputs)

        return outputs, attn_weights

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_length = 50

    pos_encodings = torch.FloatTensor(seq_length, 1, 128).uniform_(0,1)
    seq = torch.FloatTensor(seq_length, 8, 128).fill_(1.0)

    pos_encodings = pos_encodings.to(device)
    seq = seq.to(device)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=6)
    parser.add_argument('--model_dim', default=128)
    parser.add_argument('--num_heads', default=2)
    parser.add_argument('--dim_ffn', default=16)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--init_fn_name', default='xavier')
    parser.add_argument('--pre_normalization', default=True)
    args = parser.parse_args()

    encoder = TransformerEncoder(
        num_layers = args.num_layers,
        model_dim = args.model_dim,
        num_heads=args.num_heads,
        dim_ffn=args.dim_ffn,
        dropout=args.dropout,
        init_fn_name=args.init_fn_name,
        pre_normalization=args.pre_normalization
    )
    encoder.to(device)
    encoder.eval()
    
    output, attn_weights = encoder(seq, pos_encodings)