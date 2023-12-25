import torch
import torch.nn as nn

from utils.others import pose_from_vel


class PVLSTM(nn.Module):
    def __init__(self, args):
        super(PVLSTM, self).__init__()
        self.args = args
        input_size = output_size = int(args.keypoints_num * args.keypoint_dim)
        self.pose_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)
        self.vel_encoder = Encoder(input_size, args.hidden_size, args.n_layers, args.dropout_enc)
        self.vel_decoder = Decoder(args.pred_frames_num, input_size, output_size, args.hidden_size, args.n_layers,
                                   args.dropout_pose_dec, 'hardtanh', args.hardtanh_limit)

    def forward(self, inputs):
        pose = inputs['observed_pose']
        vel = pose[..., 1:, :] - pose[..., :-1, :]

        (hidden_vel, cell_vel) = self.vel_encoder(vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        (hidden_pose, cell_pose) = self.pose_encoder(pose.permute(1, 0, 2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)

        vel_dec_input = vel[:, -1, :]
        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel
        pred_vel = self.vel_decoder(vel_dec_input, hidden_dec, cell_dec)
        pred_pose = pose_from_vel(pred_vel, pose[..., -1, :])
        outputs = {'pred_pose': pred_pose, 'pred_vel': pred_vel}

        return outputs


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)

    def forward(self, inpput_):
        outputs, (hidden, cell) = self.lstm(inpput_)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, outputs_num, input_size, output_size, hidden_size, n_layers, dropout, activation_type,
                 hardtanh_limit=None):
        super().__init__()
        self.outputs_num = outputs_num
        self.dropout = nn.Dropout(dropout)
        lstms = [
            nn.LSTMCell(input_size=input_size if i == 0 else hidden_size, hidden_size=hidden_size) for
            i in range(n_layers)]
        self.lstms = nn.Sequential(*lstms)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=output_size)
        if activation_type == 'hardtanh':
            self.activation = nn.Hardtanh(min_val=-1 * hardtanh_limit, max_val=hardtanh_limit, inplace=False)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, inputs, hiddens, cells):
        dec_inputs = self.dropout(inputs)
        if len(hiddens.shape) < 3 or len(cells.shape) < 3:
            hiddens = torch.unsqueeze(hiddens, 0)
            cells = torch.unsqueeze(cells, 0)
        device = 'cuda' if inputs.is_cuda else 'cpu'
        outputs = torch.tensor([], device=device)
        for j in range(self.outputs_num):
            for i, lstm in enumerate(self.lstms):
                if i == 0:
                    hiddens[i], cells[i] = lstm(dec_inputs, (hiddens.clone()[i], cells.clone()[i]))
                else:
                    hiddens[i], cells[i] = lstm(hiddens.clone()[i - 1], (hiddens.clone()[i], cells.clone()[i]))
            output = self.activation(self.fc_out(hiddens.clone()[-1]))
            dec_inputs = output.detach()
            outputs = torch.cat((outputs, output.unsqueeze(1)), 1)
        return outputs
