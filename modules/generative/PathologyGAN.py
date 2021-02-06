from typing import no_type_check
from tqdm import tqdm
from torch.nn.modules import loss
from .Mapping import Mapping
from .Discriminator import DiscriminatorResnet
from .Generator import Generator
import torch
from torch import optim, nn
from dataset import Dataset
from itertools import chain
from modules.regularization.OrthogonalRegularizer import OrthogonalRegularizer
from modules.utils import output_sample_image_grid

# Should we extend nn.Module here?
# I don't think it's necessary now, extended it anyway to access self.state_dict

class PathologyGAN(nn.Module):
    def __init__(
            self,
            dataset: Dataset,
            learning_rate_d: float,
            learning_rate_g: float,
            beta_1: float,
            beta_2: float,
            epochs: int,
            z_dim: int,
            checkpoint_path: str,
            gp_coeff: float,
            output_path: str
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.build_model()
        self.initialize_optimizers(
            learning_rate_d, learning_rate_g, beta_1, beta_2)
        self.regularizer = OrthogonalRegularizer(1e-4)

        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.epochs = epochs
        self.z_dim = z_dim
        self.gp_coeff = gp_coeff

        # Torch equivalent for tf.nn.sigmoid_cross_entropy_with_logits
        self.multi_label_margin_loss = nn.MultiLabelSoftMarginLoss()
        # We can register buffers here to track variables,too

    def build_model(self) -> None:
        self.mapping = Mapping().to(self.device)
        self.disc = DiscriminatorResnet().to(self.device)
        self.gen = Generator().to(self.device)

    def initialize_optimizers(self, learning_rate_d: float, learning_rate_g: float, beta_1: float, beta_2: float) -> None:
        disc_parameters = self.disc.parameters()
        # Generator and Mapping are updated with the same optimizer
        gen_parameters = chain(self.mapping.parameters(),
                               self.gen.parameters())
        self.optimizerD = optim.Adam(
            disc_parameters, lr=learning_rate_d, betas=(beta_1, beta_2))
        self.optimizerG = optim.Adam(
            gen_parameters, lr=learning_rate_g, betas=(beta_1, beta_2))

    def load_weights(self) -> None:
        self.load_state_dict(self.checkpoint_path)

    def store_weights(self) -> None:
        torch.save(self.state_dict(), self.checkpoint_path)

    # Output images after each epoch with sampling from z
    def generate_sample_images(self, latent_input: torch.Tensor, epoch: int) -> torch.Tensor:
        w = self.mapping(latent_input)
        images = self.gen(w,w).detach()
        output_sample_image_grid(images, grid_size = 6, output_path=self.output_path, epoch=epoch)
        return images

    def get_gradient_penalty(self, epsilon: torch.Tensor, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        mixed_images = real_images * epsilon + fake_images * (1 - epsilon)

        # Calculate the critic's scores on the mixed images
        mixed_scores = self.disc(mixed_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            # Note: You need to take the gradient of outputs with respect to inputs.
            # This documentation may be useful, but it should not be necessary:
            # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
            #### START CODE HERE ####
            inputs=mixed_images,
            outputs=mixed_scores,
            #### END CODE HERE ####
            # These other parameters have to do with the pytorch autograd engine works
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1)**2)

        return penalty

    def train_discriminator(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:

        self.disc.zero_grad()
        out_real = self.disc(real_images)
        out_fake = self.disc(fake_images.detach())

        # Relativistic Loss, take the diff of reals and fakes
        real_fake_diff = out_real - torch.mean(out_fake, dim=0, keepdim=True)
        fake_real_diff = out_fake - torch.mean(out_real, dim=0, keepdim=True)

        loss_dis_real = self.multi_label_margin_loss(
            real_fake_diff, torch.ones(*real_fake_diff.shape))
        loss_dis_fake = self.multi_label_margin_loss(
            fake_real_diff, torch.zeros(*real_fake_diff.shape))

        # Gradient Penalty Loss
        epsilon = torch.rand(len(real_images), 1, 1, 1,
                             device=self.device, requires_grad=True)
        grad_penalty = self.get_gradient_penalty(
            real_images, fake_images.detach(), epsilon)

        # Orthogonality Loss
        orthogonality_loss = self.regularizer.get_regularizer_loss(self.disc)
        loss_discriminator = loss_dis_fake + loss_dis_real + \
            orthogonality_loss + grad_penalty * self.gp_coeff
        loss_discriminator.backward()

        self.optimizerD.step()

        return loss_discriminator.item()

    # The generator basically has the inverse loss of disc
    def train_generator(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        self.gen.zero_grad()
        self.mapping.zero_grad()
        out_real = self.disc(real_images)
        out_fake = self.disc(fake_images)

        # Relativistic Loss
        real_fake_diff = out_real - torch.mean(out_fake, dim=0, keepdim=True)
        fake_real_diff = out_fake - torch.mean(out_real, dim=0, keepdim=True)

        loss_gen_real = self.multi_label_margin_loss(
            fake_real_diff, torch.ones(*real_fake_diff.shape))
        loss_gen_fake = self.multi_label_margin_loss(
            real_fake_diff, torch.zeros(*real_fake_diff.shape))

        # Torch does not implicitly regularize, so we need to add to the total loss
        orthogonality_loss = self.regularizer.get_regularizer_loss(
            self.gen) + self.regularizer.get_regularizer_loss(self.mapping)

        loss_generator = loss_gen_fake + loss_gen_real + orthogonality_loss
        loss_generator.backward()

        self.optimizerG.step()

        return loss_generator.item()

    def train(self, restore: bool = False, n_critic: int = 5) -> None:
        iters = 0

        # Create a steady latent input here to benchmark epochs
        steady_latent = torch.randn((36, self.z_dim)).to(self.device)

        if restore:
            self.load_weights()

        print("Starting Training Loop")
        for epoch in range(self.epochs):
            pbar = tqdm(total=self.dataset.size)
            pbar.set_description(f"Epoch {epoch}")
            for images, labels in self.dataset:

                z_latent = torch.randn(
                    len(images), self.z_dim, device=self.device)

                batch_images = images.to(self.device)

                w_latent = self.mapping(z_latent)
                fake_images = self.gen(w_latent, w_latent)

                loss_disc = self.train_discriminator(batch_images, fake_images)

                # Train generator every n steps
                if iters % n_critic == 0:
                    loss_gen = self.train_generator(batch_images, fake_images)
                    pbar.set_postfix_str({"Loss Disc": loss_disc, "Loss Gen": loss_gen})
                iters += 1
                pbar.update(self.dataset.i)

            self.store_weights()
            self.dataset.reset()
            self.generate_sample_images(steady_latent, epoch)

        print("Training Finished")
