from torch.nn import Module
import torch
from models.pgbig import base_model as BaseBlock
from models.pgbig.data_proc import Preprocess, Postprocess, Human36m_Postprocess, Human36m_Preprocess, AMASS_3DPW_Postprocess, AMASS_3DPW_Preprocess

from models.pgbig import util

"""
在model1的基础上添加st_gcn,修改 bn 
"""

class MultiStageModel(Module):
    def __init__(self, opt):
        super(MultiStageModel, self).__init__()

        self.opt = opt
        self.kernel_size = opt.kernel_size
        self.d_model = opt.d_model
        # self.seq_in = seq_in
        self.dct_n = opt.dct_n
        # ks = int((kernel_size + 1) / 2)
        assert opt.kernel_size == 10

        self.in_features = opt.in_features
        self.num_stage = opt.num_stage
        self.node_n = self.in_features//3

        self.encoder_layer_num = 1
        self.decoder_layer_num = 2

        self.input_n = opt.obs_frames_num
        self.output_n = opt.pred_frames_num

        self.gcn_encoder1 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                               node_n=self.node_n,
                                               seq_len=self.dct_n,
                                               p_dropout=opt.drop_out,
                                               num_stage=self.encoder_layer_num)

        self.gcn_decoder1 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                               node_n=self.node_n,
                                               seq_len=self.dct_n*2,
                                               p_dropout=opt.drop_out,
                                               num_stage=self.decoder_layer_num)

        self.gcn_encoder2 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder2 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

        self.gcn_encoder3 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder3 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

        self.gcn_encoder4 = BaseBlock.GCN_encoder(in_channal=3, out_channal=self.d_model,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.encoder_layer_num)

        self.gcn_decoder4 = BaseBlock.GCN_decoder(in_channal=self.d_model, out_channal=3,
                                                 node_n=self.node_n,
                                                 seq_len=self.dct_n * 2,
                                                 p_dropout=opt.drop_out,
                                                 num_stage=self.decoder_layer_num)

    def forward(self, src, input_n=10, output_n=10, itera=1):
        output_n = self.output_n
        input_n = self.input_n

        bs = src.shape[0]
        # [2000,512,22,20]
        dct_n = self.dct_n
        idx = list(range(self.kernel_size)) + [self.kernel_size -1] * output_n
        # [b,20,66]
        input_gcn = src[:, idx].clone()

        dct_m, idct_m = util.get_dct_matrix(input_n + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(self.opt.device)
        idct_m = torch.from_numpy(idct_m).float().to(self.opt.device)

        # [b,20,66] -> [b,66,20]
        input_gcn_dct = torch.matmul(dct_m[:dct_n], input_gcn).permute(0, 2, 1)

        # [b,66,20]->[b,22,3,20]->[b,3,22,20]->[b,512,22,20]
        input_gcn_dct = input_gcn_dct.reshape(bs, self.node_n, -1, self.dct_n).permute(0, 2, 1, 3)

        #stage1
        latent_gcn_dct = self.gcn_encoder1(input_gcn_dct)
        #[b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_1 = self.gcn_decoder1(latent_gcn_dct)[:, :, :, :dct_n]

        #stage2
        latent_gcn_dct = self.gcn_encoder2(output_dct_1)
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_2 = self.gcn_decoder2(latent_gcn_dct)[:, :, :, :dct_n]

        #stage3
        latent_gcn_dct = self.gcn_encoder3(output_dct_2)
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_3 = self.gcn_decoder3(latent_gcn_dct)[:, :, :, :dct_n]

        #stage4
        latent_gcn_dct = self.gcn_encoder4(output_dct_3)
        # [b,512,22,20] -> [b, 512, 22, 40]
        latent_gcn_dct = torch.cat((latent_gcn_dct, latent_gcn_dct), dim=3)
        output_dct_4 = self.gcn_decoder4(latent_gcn_dct)[:, :, :, :dct_n]

        output_dct_1 = output_dct_1.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_2 = output_dct_2.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_3 = output_dct_3.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)
        output_dct_4 = output_dct_4.permute(0, 2, 1, 3).reshape(bs, -1, dct_n)

        # [b,20 66]->[b,20 66]
        output_1 = torch.matmul(idct_m[:, :dct_n], output_dct_1.permute(0, 2, 1))
        output_2 = torch.matmul(idct_m[:, :dct_n], output_dct_2.permute(0, 2, 1))
        output_3 = torch.matmul(idct_m[:, :dct_n], output_dct_3.permute(0, 2, 1))
        output_4 = torch.matmul(idct_m[:, :dct_n], output_dct_4.permute(0, 2, 1))

        return output_4, output_3, output_2, output_1


class PGBIG(Module):
    def __init__(self, args):
        super(PGBIG, self).__init__()
        self.args = args
        self.in_n = args.obs_frames_num
        self.out_n = args.pred_frames_num

        if args.pre_post_process == 'human3.6m':
            self.preprocess = Human36m_Preprocess(args).to(args.device)
            self.postprocess = Human36m_Postprocess(args).to(args.device)
        elif args.pre_post_process == 'AMASS' or args.pre_post_process == '3DPW':
            self.preprocess = AMASS_3DPW_Preprocess(args).to(args.device)
            self.postprocess = AMASS_3DPW_Postprocess(args).to(args.device)
        else:
            self.preprocess = Preprocess(args).to(args.device)
            self.postprocess = Postprocess(args).to(args.device)

        for p in self.preprocess.parameters():
            p.requires_grad = False

        for p in self.postprocess.parameters():
            p.requires_grad = False
        
        self.net = MultiStageModel(args).to(args.device)


    def forward(self, batch):
        observed_data = batch["observed_pose"].to(self.args.device)
        observed_data = self.preprocess(observed_data, normal=True)
        p4, p3, p2, p1 = self.net(observed_data, input_n=self.in_n, output_n=self.out_n, itera=1)
        
        return {
            'pred_pose': self.postprocess(batch['observed_pose'], p4[:, self.in_n:, :], normal=True),
            'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4
        }
        

