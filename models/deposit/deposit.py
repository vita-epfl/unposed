import math
import torch
from torch import nn
from .diffusion_util import diff_CSDI
import numpy as np

class DePOSit(nn.Module):
    def __init__(self, args):
        super(DePOSit, self).__init__()
        self.args = args
        self.config = args.config
        self.device = args.device
        self.target_dim = args.target_dim

        self.emb_time_dim = self.config["model"]["timeemb"]
        self.emb_feature_dim = self.config["model"]["featureemb"]
        self.is_unconditional = self.config["model"]["is_unconditional"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = self.config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        elif config_diff["schedule"] == "cosine":
            self.beta = self.betas_for_alpha_bar(
                self.num_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.5):
        # """
        # Create a beta schedule that discretizes the given alpha_t_bar function,
        # which defines the cumulative product of (1-beta) over time from t = [0,1].
        # :param num_diffusion_timesteps: the number of betas to produce.
        # :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
        #                   produces the cumulative product of (1-beta) up to that
        #                   part of the diffusion process.
        # :param max_beta: the maximum beta to use; use values lower than 1 to
        #                  prevent singularities.
        # """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
            self, observed_data, cond_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
            self, observed_data, cond_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t].to(self.device)  # (B,1,1)
        noise = torch.randn_like(observed_data.cpu()).to(self.device)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = 1 - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        # return loss
        return loss.mean()

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs.cpu())
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data.cpu())

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample.cpu())
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = (current_sample * (1 - cond_mask) + observed_data * cond_mask).detach()
        return imputed_samples

    def impute_ddim(self, observed_data, cond_mask, side_info, _):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, 1, K, L).to(self.device)

        current_sample = torch.randn_like(observed_data.cpu()).to(self.device)  # Initial random sample
        for t in range(self.num_steps - 1, -1, -1):
        
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
            diff_input = torch.cat([cond_obs, noisy_target], dim=1)

            predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
            
            # Apply DDIM step here instead of DDPM's noise application
            current_sample = self.deterministic_update_function(current_sample, predicted, t)

        imputed_samples[:, 0] = (current_sample * (1 - cond_mask) + observed_data * cond_mask).detach()
        return imputed_samples

    def deterministic_update_function(self, state, pred_noise, t):
        at = self.alpha[t]
        at_minus_1 = self.alpha[t-1] if t > 0 else self.alpha[0]
        pred_x0 = ((state - (1.0 - at) ** 0.5 * pred_noise) / at ** 0.5)
        if t == 0:
            return pred_x0
        xt_prev = at_minus_1 ** 0.5 * pred_x0 + (1.0 - at_minus_1) ** 0.5 * pred_noise
        return xt_prev

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_tp,
            gt_mask
        ) = self.process_data(batch)

        cond_mask = gt_mask

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        pred_pose = self.evaluate(batch, 1)[0]
        pred_pose = pred_pose[:, 0, :, -batch['future_pose'].shape[1]:]
        pred_pose = pred_pose.transpose(1, 2)

        return {
            'loss': loss_func(observed_data, cond_mask, side_info, is_train),
            'pred_pose': pred_pose
        }

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_tp,
            gt_mask
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = 1 - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            impute_methods = {"ddpm": self.impute, "ddim": self.impute_ddim}
            diffusion_type = self.config["diffusion"]["type"].lower()
            impute_method = impute_methods.get(diffusion_type, self.impute)
            samples = impute_method(observed_data, cond_mask, side_info, n_samples)
        return samples, observed_data, target_mask, observed_tp

    def process_data(self, batch):
        batch = self.map_data(batch)
        pose = batch["pose"].to(self.device).float()
        tp = batch["timepoints"].to(self.device).float()
        mask = batch["mask"].type(torch.float32).to(self.device).float()

        pose = pose.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)

        return (
            pose,
            tp,
            mask
        )

    def map_data(self, batch):
        in_n = batch['observed_pose'].shape[1]
        out_n = batch['future_pose'].shape[1]

        combined_poses = torch.cat((batch['observed_pose'], batch['future_pose']), dim=1)
        B = batch['observed_pose'].shape[0]

        # no_miss mask
        mask = np.zeros((in_n + out_n, batch['observed_pose'].shape[2])) # (in + out, joints * dims)
        mask[0:in_n, :] = 1
        mask[in_n:in_n + out_n, :] = 0
        mask = np.tile(mask, (B, 1, 1))

        timepoints = np.arange(in_n + out_n)
        timepoints = np.tile(timepoints, (B, 1))
        
        return {
            "pose": combined_poses / 1000.0,
            "timepoints": torch.tensor(timepoints),
            "mask": torch.tensor(mask)
        }