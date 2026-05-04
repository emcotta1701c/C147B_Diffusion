import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionalFM(nn.Module):
    def __init__(self, modelconfig):
        super().__init__()
        self.modelconfig = modelconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(
            self.modelconfig.num_channels,
            self.modelconfig.num_feat,
            self.modelconfig.num_classes,
            self.modelconfig.input_dim,
        )

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

        # pass

        B = images.shape[0]
        x_0 = np.randn_like(images)
        cemb = F.one_hot(conditions, num_classes=self.modelconfig.num_classes).float()
        random_val = torch.rand(B, 1, device=device)
        mask = (random_val < self.modelconfig.mask_p).float()
        cemb = cemb * (1 - mask) + mask * self.modelconfig.condition_mask_value
        t = np.rand(B)
        x_t = (1-t) * x_0 + t * images
        u_t = images - x_0
        loss = self.loss_fn(self.network(x_t, t, cemb) - u_t)

        # ==================================================== #
        return loss

    def sample(self, conditions, omega):
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

        # pass
        
        B = conditions.shape[0]
        output_shape = (B, self.modelconfig.num_channels, self.modelconfig.input_dim, self.modelconfig.input_dim)
        x = np.randn(output_shape)
        delta_t = 1 / self.modelconfig.T
        for k in range(0, self.modelconfig.T):
            t_k = torch.full((B,1), k * delta_t)
            v_t = (1 + omega) * self.network(x, t_k, conditons) - omega * self.network(x, t_k)
            x = x + delta_t * v_t


        # ==================================================== #
        generated_images = (x * 0.3081 + 0.1307).clamp(0, 1)
        return generated_images
