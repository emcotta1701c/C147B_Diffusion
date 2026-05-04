import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels, 
            self.modelconfig.num_feat, 
            self.modelconfig.num_classes, 
            self.modelconfig.input_dim
        )

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.modelconfig.beta_1, self.modelconfig.beta_T, self.modelconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       
        
        t_s = t_s.squeeze()
        beta_t = torch.linspace(beta_1, beta_T, T)
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        oneover_sqrt_alpha = 1 / torch.sqrt(alpha_t)
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)
        
        # Convert from precomputed schedule to timestep specific schedule
        beta_t = beta_t[t_s]
        sqrt_beta_t = sqrt_beta_t[t_s]
        alpha_t = alpha_t[t_s]
        oneover_sqrt_alpha = oneover_sqrt_alpha[t_s]
        alpha_t_bar = alpha_t_bar[t_s]
        sqrt_alpha_bar = sqrt_alpha_bar[t_s]
        sqrt_oneminus_alpha_bar = sqrt_oneminus_alpha_bar[t_s]

        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function.  

        B = images.shape[0]
        cemb = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float()
        random_val = torch.rand(B, 1, device=device)
        mask = (random_val < self.modelconfig.mask_p).float()
        cemb = cemb * (1 - mask) + mask * self.modelconfig.condition_mask_value
        t = torch.randint(1, self.modelconfig.T + 1, (B,)).float()
        t = t / self.modelconfig.T  # normalize to [0,1]
        eps = torch.randn_like(sample)
        scheduler_dict = self.scheduler(t)
        sqrt_alpha_bar = scheduler_dict['sqrt_alpha_bar']
        sqrt_oneminus_alpha_bar = scheduler_dict['sqrt_oneminus_alpha_bar']
        x_t = sqrt_alpha_bar * sample + sqrt_oneminus_alpha_bar * eps
        noise_loss = self.loss_fn(self.network(x_t, t, cemb_sample) - eps)
        
        # pass



        # ==================================================== #
        return noise_loss

    def sample(self, conditions, omega):
        T = self.modelconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        B = conditions.shape[0]
        X_T = torch.randn(B, self.modelconfig.num_channels, self.modelconfig.input_dim, self.modelconfig.input_dim)
        for t in range(T, 0, -1):
            t_batch = torch.full((B, 1), t)
            t_normalized = t_batch / self.modelconfig.T
            z = torch.zeros_like(X_T)
            if t > 1:
                z = torch.randn_like(X_T)
            eps_t = (1+omega) * self.network(X_T, t_normalized, conditions) - omega * self.network(x_T, t_normalized)
            schedule = self.scheduler(t_batch)
            
            oneover_sqrt_alpha = schedule['oneover_sqrt_alpha']
            oneminus_alpha = 1 - schedule['alpha_t']
            sqrt_oneminus_alpha_bar = schedule['sqrt_oneminus_alpha_bar']
            sigma_t = schedule['sqrt_beta_t']
            x_tminus1 = schedule['oneover_sqrt_alpha'] * (X_T - oneminus_alpha / sqrt_oneminus_alpha_bar * eps_t) + sigma_t * z
            
            X_T = x_tminus1
            
        # pass


        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1)
        return generated_images
