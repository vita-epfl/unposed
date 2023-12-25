
import numpy as np
import torch
import torch.nn as nn


class PUALoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        '''
        args mush have:
        @param init_mean : float : initialize S such that the mean of S is init_mean, 3.5 is a good default
        @param tasks : str : list of tasks as a string, J for tasks over joints, T for over time(frames), A for actions. if A is used, 'action' must be in the input.
        @param nT: int: number of frames to predict
        @param nJ: int: number of joints
        @param action_list : list(str) : name of different actions as a list of str. used in case of A present in tasks.
        @param time_prior: str : what time prior to use, must be one of sig5, sig*, none
        @param clipMinS, clipMaxS: float : these values are used to slip s. MinS is needed if there are tasks in the input with errors near zero. one can set them to None, resulting in no cliping.
        @param device : str : device to run torch on
        '''
        self.args = args
        
        init_mean = self.args.init_mean
        self.s = torch.ones(1, 1, 1, requires_grad = True).to(self.args.device) * init_mean
        # Fix tasks for joints
        if 'J' in args.tasks:
            self.nJ = args.nJ
            self.s = self.s.repeat(1, 1, self.nJ)
        else:
            self.nJ = 1
        #fix tasks for time
        if 'T' not in args.tasks:
            self.nT = 1
        elif args.time_prior == 'sig5':  
            self.nT = 5
            self.s = self.s.repeat(1, 5, 1)
            self.s[:, :, :] = 0
            self.s[:, 0, :] = init_mean
            self.s[:, 2, :] = 1
        elif args.time_prior == 'sig*':
            self.nT = 3
            self.s = self.s.repeat(1, 3, 1)
            self.s[:, 0, :] = init_mean
            self.s[:, 1, :] = 1
            self.s[:, 2, :] = -10
        elif args.time_prior == 'none':
            self.nT = args.nT
            self.s = self.s.repeat(1, self.nT, 1)
        elif 'poly' in args.time_prior:
            self.nT = int(args.time_prior[4:]) + 1
            self.s = self.s.repeat(1, self.nT, 1)
            self.s[:, 1:, :] = 0
        else:
            raise Exception("{} is not a supported prior for time axis, options are: [sig5, sig*, none].".format(args.time_prior))
        # fix tasks for action
        if 'A' in args.tasks:
            self.action_list = args.action_list
            self.nA = len(self.action_list)
            self.action_map = {self.action_list[i]: i for i in range(self.nA)}
            self.s = self.s.repeat(self.nA, 1, 1)
            self.sigma = nn.Embedding(self.nA, self.nT * self.nJ)
            self.sigma.weight = nn.Parameter(self.s.view(-1, self.nT * self.nJ))
        else:
            self.nA = None
            self.sigma = nn.Parameter(self.s)

    def calc_sigma(self, y_true):
        local_sigma = self.sigma
        if self.nA is not None:
            actions = y_true['action']
            indx = torch.tensor([self.action_map[act] for act in actions]).to(self.args.device)
            local_sigma = local_sigma(indx)
            local_sigma = local_sigma.view(-1, self.nT, self.nJ)
        
        if 'T' in self.args.tasks:
            if self.args.time_prior == 'sig5':
                x = torch.arange(self.args.nT).to(self.args.device).unsqueeze(1).unsqueeze(0) # 1, T, 1
                c = 2 * local_sigma[:, 3 - 1, :] * local_sigma[:, 5 - 1, :] / torch.abs(local_sigma[:, 3 - 1, :] + local_sigma[:, 5 - 1, :])
                f = 1 / (1 + torch.exp(-c * (local_sigma[:, 4 - 1, :] - x)))
                g = torch.exp(local_sigma[:, 3 - 1, :] * (local_sigma[:, 4 - 1, :] - x))
                h = torch.exp(local_sigma[:, 5 - 1, :] * (local_sigma[:, 4 - 1, :] - x))
                local_sigma = local_sigma[:, 1 - 1, :] + (local_sigma[:, 2 - 1, :] / (1 + f * g + (1 - f) * h))
                
            elif self.args.time_prior == 'sig*':
                x = torch.arange(self.args.nT).to(self.args.device).unsqueeze(1).unsqueeze(0) # 1, T, 1
                local_sigma = local_sigma[:, 0:1, :] / (1 + torch.exp(local_sigma[:, 1:2, :] * (local_sigma[:, 2:3, :] - x)))
            elif 'poly' in self.args.time_prior:
                x = torch.arange(self.args.nT).to(self.args.device).unsqueeze(1).unsqueeze(1).unsqueeze(0) / 10 # 1, T, 1, 1
                po = torch.arange(self.nT).to(self.args.device).unsqueeze(1).unsqueeze(0).unsqueeze(0) # 1, 1, D, 1
                x = x ** po # 1, T, D, 1
                local_sigma = local_sigma.unsqueeze(1) # 1, 1, D, ?
                local_sigma = (local_sigma * x).sum(dim=-2) # 1, T, ?
                
                
        local_sigma = torch.clamp(local_sigma, min=self.args.clipMinS, max=self.args.clipMaxS)
        
        return local_sigma #local_sigma
    
        

    def forward(self, y_pred, y_true):
      
        sigma = self.calc_sigma(y_true)

        y_pred = y_pred['pred_pose'] # B,T,JC
        y_true = y_true['future_pose'] # B,T,JC

        B,T,JC = y_pred.shape
        assert T == self.args.nT and JC % self.args.nJ == 0, "Either number or predicted frames (nT) is not right, or number of joints * dim of each joint is not dividable by nJ"
        J = self.args.nJ
        C = JC // J

        y_pred = y_pred.view(B, T, J, C)
        y_true = y_true.view(B, T, J, C)

        l = torch.norm(y_pred - y_true, dim=-1) # B,T,J
        l = torch.mean(torch.exp(-sigma) * l + sigma)

        return {
          'loss' : l
        }