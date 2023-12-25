import torch
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np
import math
import re

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m
class HistoryRepeatsItself(nn.Module):
    def __init__(self, args):
        super(HistoryRepeatsItself, self).__init__()
        args.loss.itera = args.itera
        args.loss.un_mode = args.un_mode
        self.modality = args.modality
        args.loss.modality = self.modality

        self.args = args
        self.init_mode = args.init_mode
        print(args)
        self.device = args.device
        self.net_pred = AttModel(in_features=args.in_features, kernel_size=args.kernel_size, d_model=args.d_model,
                                 num_stage=args.num_stage, dct_n=args.dct_n, device=self.device)

        l_p3d = 0

        self.in_n = args.input_n
        self.out_n = args.output_n
        if args.un_mode == 'sig5-TJPrior':
            un_params = torch.nn.Parameter(torch.zeros((args.in_features//3 + args.output_n + args.kernel_size, 5)))
        elif 'sig5' in args.un_mode:
            un_params = torch.nn.Parameter(torch.zeros(args.in_features//3, 5))
        elif 'sigstar' in args.un_mode:
            un_params = torch.nn.Parameter(torch.zeros(args.in_features//3, 2))
        elif 'poly' in args.un_mode:
            try:
                params_count = int(re.findall("\d+", args.un_mode)[0]) + 2
                self.params_count = params_count
            except:
                assert False, "you must have a number after 'poly'"
            un_params = torch.nn.Parameter(torch.zeros(args.in_features//3, params_count))
        else:
            if args.itera == 1:
                un_params = torch.nn.Parameter(torch.zeros(15, self.out_n + args.kernel_size ,args.in_features//3))
            else:
                un_params = torch.nn.Parameter(torch.zeros(15, 10 + args.kernel_size ,args.in_features//3))

        self.un_params = un_params
        if self.init_mode == "descending":
            torch.nn.init.constant_(self.un_params[:, 0], -0.2)
            torch.nn.init.constant_(self.un_params[:, 1], 3.7)
            torch.nn.init.constant_(self.un_params[:, 2], -0.2)
            torch.nn.init.constant_(self.un_params[:, 3], 10)
            torch.nn.init.constant_(self.un_params[:, 4], -0.1)

        elif self.init_mode == "increasing":
            torch.nn.init.constant_(self.un_params[:, 0], 0)
            torch.nn.init.constant_(self.un_params[:, 1], 3)
            torch.nn.init.constant_(self.un_params[:, 2], 0.2)
            torch.nn.init.constant_(self.un_params[:, 3], 10.7)
            torch.nn.init.constant_(self.un_params[:, 4], 0.1)
            
        elif self.init_mode == "constant-one":
            torch.nn.init.constant_(self.un_params[:, 0], 1)
            torch.nn.init.constant_(self.un_params[:, 1], 0) # this is not a bug :)
            torch.nn.init.constant_(self.un_params[:, 2], 1)
            torch.nn.init.constant_(self.un_params[:, 3], 1)
            torch.nn.init.constant_(self.un_params[:, 4], 1)

        elif self.init_mode == "increasing1":
            torch.nn.init.constant_(self.un_params[:, 0], 0)
            torch.nn.init.constant_(self.un_params[:, 1], 7.8)
            torch.nn.init.constant_(self.un_params[:, 2], 0.5)
            torch.nn.init.constant_(self.un_params[:, 3], 17.8)
            torch.nn.init.constant_(self.un_params[:, 4], 0.2)

        elif self.init_mode == "increasing2":
            torch.nn.init.constant_(self.un_params[:, 0], 2.1)
            torch.nn.init.constant_(self.un_params[:, 1], 2.6)
            torch.nn.init.constant_(self.un_params[:, 2], 0.5)
            torch.nn.init.constant_(self.un_params[:, 3], 17.8)
            torch.nn.init.constant_(self.un_params[:, 4], 0.2)

        elif self.init_mode == "increasing3":
            torch.nn.init.constant_(self.un_params[:, 0], 2.1)
            torch.nn.init.constant_(self.un_params[:, 1], 6)
            torch.nn.init.constant_(self.un_params[:, 2], 0.5)
            torch.nn.init.constant_(self.un_params[:, 3], 17.8)
            torch.nn.init.constant_(self.un_params[:, 4], 0.2)

        elif self.init_mode == "increasing4":
            torch.nn.init.constant_(self.un_params[:, 0], 0.6)
            torch.nn.init.constant_(self.un_params[:, 1], 4.7)
            torch.nn.init.constant_(self.un_params[:, 2], 0.1)
            torch.nn.init.constant_(self.un_params[:, 3], 20)
            torch.nn.init.constant_(self.un_params[:, 4], 0.2)
        elif self.init_mode == 'poly_decreasing':
            coeff = 1
            for i in range(self.params_count):
                torch.nn.init.constant_(self.un_params[:, i], coeff)
                coeff /= 10

        elif bool(re.findall(r'^default_[-+]?(?:\d*\.\d+|\d+)_[-+]?(?:\d*\.\d+|\d+)$', self.init_mode)):
            mean, std = [float(n) for n in re.findall('[-+]?(?:\d*\.\d+|\d+)', self.init_mode)]
            torch.nn.init.normal_(self.un_params, mean=mean, std=std)

        elif self.init_mode == "default":
            mean, std = 0, 1
            torch.nn.init.normal_(self.un_params, mean=mean, std=std)
        else:
            raise Exception("The defined init mode is not supported.")

        print(self.un_params)
        if self.modality == "Human36":
            self.dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        elif self.modality == "AMASS" or self.modality == "3DPW":
            pass
        else:
            assert False, "The modality is not supported."
        self.seq_in = args.kernel_size
        self.sample_rate = 2
        self.joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        self.index_to_ignore = np.concatenate(
            (self.joint_to_ignore * 3, self.joint_to_ignore * 3 + 1, self.joint_to_ignore * 3 + 2))
        self.joint_equal = np.array([13, 19, 22, 13, 27, 30])
        self.index_to_equal = np.concatenate((self.joint_equal * 3, self.joint_equal * 3 + 1, self.joint_equal * 3 + 2))
        self.itera = args.itera
        self.idx = np.expand_dims(np.arange(self.seq_in + self.out_n), axis=1) + (
                self.out_n - self.seq_in + np.expand_dims(np.arange(self.itera), axis=0))

    def forward_human(self, inputs):
        seq = torch.cat((inputs['observed_pose'], inputs['future_pose']), dim=1)
        p3d_h36 = seq.reshape(seq.shape[0], seq.shape[1], -1)
        batch_size, seq_n, _ = p3d_h36.shape
        p3d_h36 = p3d_h36.float() 
        p3d_sup = p3d_h36.clone()[:, :, self.dim_used][:, -self.out_n - self.seq_in:].reshape(
            [-1, self.seq_in + self.out_n, len(self.dim_used) // 3, 3])
        p3d_src = p3d_h36.clone()[:, :, self.dim_used]
        if self.itera == 1:
            p3d_out_all = self.net_pred(p3d_src, input_n=self.in_n, output_n=self.out_n, itera=self.itera)
            p3d_out = p3d_h36.clone()[:, self.in_n:self.in_n + self.out_n]
            p3d_out[:, :, self.dim_used] = p3d_out_all[:, self.seq_in: self.seq_in + self.out_n, 0]
            p3d_out[:, :, self.index_to_ignore] = p3d_out[:, :, self.index_to_equal]
            p3d_out = p3d_out.reshape([-1, self.out_n, 96])
            p3d_out_all = p3d_out_all.reshape(
                [batch_size, self.seq_in + self.out_n, self.itera, len(self.dim_used) // 3, 3])
        else:
            if self.training:
                iterr = 1
                out_ = 10
            else:
                iterr = self.itera
                out_ = self.out_n
            p3d_out_all = self.net_pred(p3d_src, input_n=self.in_n, output_n=10, itera=iterr)
            p3d_1 = p3d_out_all[:, :self.seq_in, 0].clone()
            p3d_out_all = p3d_out_all[:, self.seq_in:].transpose(1, 2).reshape([batch_size, 10 * iterr, -1])[:, :out_] 
            zero_ = torch.zeros_like(p3d_out_all)
            if self.training:
                p3d_out_all = torch.cat((p3d_out_all, zero_, zero_), dim=1)[:, :self.out_n]
            p3d_out = p3d_h36.clone()[:, self.in_n:self.in_n + self.out_n]
            p3d_out[:, :, self.dim_used] = p3d_out_all
            p3d_out[:, :, self.index_to_ignore] = p3d_out[:, :, self.index_to_equal]
            p3d_out = p3d_out.reshape([-1, self.out_n, 96])

            p3d_h36 = p3d_h36[:, :self.in_n + out_].reshape([-1, self.in_n + out_, 32, 3])

            p3d_out_all = torch.cat((p3d_1, p3d_out_all), dim=1)
            p3d_out_all = p3d_out_all.reshape(
                [batch_size, self.seq_in + self.out_n, len(self.dim_used) // 3, 3])
        return {'pred_pose': p3d_out_all, 'pred_metric_pose': p3d_out, 'un_params': self.un_params}
    
    def forward_amass(self,inputs):
        seq = torch.cat((inputs['observed_pose'], inputs['future_pose']), dim=1)
        bs, seq_n, joints = seq.shape
        p3d_h36 = seq.reshape(seq.shape[0], seq.shape[1], -1)

        batch_size, seq_n, _ = p3d_h36.shape
        p3d_h36 = p3d_h36.float()

        p3d_src = p3d_h36.clone()

        if self.itera == 1:
            p3d_out_all = self.net_pred(p3d_src, output_n=self.out_n, input_n=self.in_n, itera=self.itera)
            p3d_out = p3d_out_all[:, self.seq_in:].reshape([batch_size, self.out_n, joints])
            p3d_out_all = p3d_out_all[:, :, 0].reshape([batch_size, self.seq_in + self.out_n, joints//3, 3])
            
        else:
            assert False, "itera > 1 is not available for amass dataset"
        return {'pred_pose': p3d_out_all, 'pred_metric_pose': p3d_out, 'un_params': self.un_params}

    def forward(self,inputs):
        if self.modality == "Human36":
            return self.forward_human(inputs)
        elif self.modality == "AMASS" or self.modality == "3DPW":
            return self.forward_amass(inputs)
        else:
            assert False, "Unknown modality"
class AttModel(nn.Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10, device='cpu'):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n
        self.device = device
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                       num_stage=num_stage,
                       node_n=in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n
        src = src[:, :input_n]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(self.device)
        idct_m = torch.from_numpy(idct_m).float().to(self.device) 

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1]) 

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)
        for i in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])

            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            dct_out_tmp = self.gcn(dct_in_tmp)
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))
            if itera > 1:
                out_tmp = out_gcn.clone()[:, 0 - output_n:]
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)

                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        outputs = torch.cat(outputs, dim=2)
        return outputs


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y
