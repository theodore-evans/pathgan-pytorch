# %%
import os
import argparse
from modules.generative.PathologyGAN import PathologyGAN
from dataset import Dataset


parser = argparse.ArgumentParser(description='PathologyGAN trainer.')
parser.add_argument('--epochs', dest='epochs', type=int, default=45,
                    help='Number epochs to run: default is 45 epochs.')
parser.add_argument('--batch_size', dest='batch_size', type=int,
                    default=64, help='Batch size, default size is 64.')
parser.add_argument('--data_path', dest='data_path', type=str,
                    default=None, help='Path for the Dataset vgh_nki')
parser.add_argument('--out_path', dest='out_path', type=str,
                    default=None, help='Path for storing sample images, losses, etc')
parser.add_argument('--weights', '-w', dest='weights',
                    type=str, default=None, help='Path for model weights')
args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size

# To overcome no locks available error
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Dataset information.
dataset_path = args.data_path
data_out_path = args.out_path
image_width = 224
image_height = 224
image_channels = 3

# Hyperparameters.
learning_rate_g = 1e-4
learning_rate_d = 1e-4
beta_1 = 0.5
beta_2 = 0.9
restore = False

# Model
layers = 5
z_dim = 200
alpha = 0.2
n_critic = 5
gp_coeff = .65
loss_type = 'relativistic gradient penalty'

data = Dataset(dataset_path, image_height, image_width,
               image_channels, batch_size, data_type='train')

# TODO: add parameters and beta
pathgan = PathologyGAN(data, learning_rate_d=learning_rate_d, learning_rate_g=learning_rate_g, beta_1=beta_1, beta_2=beta_2,
                       epochs=15, z_dim=z_dim, checkpoint_path=args.weights, gp_coeff=gp_coeff, output_path=data_out_path)
pathgan.train(False, 5)
