import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

import pandas as pd

from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

from src_other import ff_mnist, ff_model, data_cleaning
import wandb

import torch.nn as nn


import matplotlib.pyplot as plt



def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt

def get_input_layer_size(opt):
    if opt.input.dataset == "mnist":
        return 784
    elif opt.input.dataset == "senti":
        return 302
    elif opt.input.dataset == "cifar10" or opt.input.dataset == "cifar100":
        return 3072
    else:
        raise ValueError("Unknown dataset.")

def get_model_and_optimizer(opt):
    model = ff_model.FF_model(opt)
    if "cuda" in opt.device or "mps" in opt.device:
        model = model.to(opt.device)
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.classification_loss.parameters())
    ]

    if opt.training.optimizer == "SGD":
        
        optimizer = torch.optim.SGD(
            [
                {
                    "params": main_model_params,
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "momentum": opt.training.momentum,
                },
                {
                    "params": model.classification_loss.parameters(),
                    "lr": opt.training.downstream_learning_rate,
                    "weight_decay": opt.training.downstream_weight_decay,
                    "momentum": opt.training.momentum,
                },
            ]
        )
    elif opt.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": main_model_params,
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "betas": (opt.training.betas[0] , opt.training.betas[1]), 
                },
                {
                    "params": model.classification_loss.parameters(),
                    "lr": opt.training.downstream_learning_rate,
                    "weight_decay": opt.training.downstream_weight_decay,
                    "betas": (opt.training.betas[0] , opt.training.betas[1]), 
                },
            ]
        )
    return model, optimizer
# 784, 2000, 2000, 2000 # main params
# 6000, 10 # classification_loss params

def get_data(opt, partition):
    # dataset = ff_mnist.FF_MNIST(opt, partition)
    if opt.input.dataset == "mnist":
        if opt.model.structure == "CwC" or opt.model.structure == "AE" or opt.model.structure == "VAE" or opt.model.structure == "VFFAE" or opt.model.structure == "FFCVAE":
            dataset = ff_mnist.CwC_MNIST(opt, partition, dataset = opt.input.dataset, number_samples = opt.input.number_samples)
        else:
            dataset = ff_mnist.FF_MNIST(opt, partition, num_classes=10)
    elif opt.input.dataset == "senti":
        dataset = ff_mnist.FF_senti(opt, partition, num_classes=2)
    elif opt.input.dataset == "cifar10":
        if opt.model.structure == "CwC" or opt.model.structure == "AE" or opt.model.structure == "VAE":
            dataset = ff_mnist.CwC_CIFAR(opt, partition, dataset = opt.input.dataset, number_samples = opt.input.number_samples)
        else:
            dataset = ff_mnist.FF_CIFAR10(opt, partition, num_classes=10)
    elif opt.input.dataset == "cifar100":
        if opt.model.structure == "CwC" or opt.model.structure == "AE" or opt.model.structure == "VAE":
            dataset = ff_mnist.CwC_CIFAR(opt, partition, dataset = opt.input.dataset, number_samples = opt.input.number_samples)
        else:
            dataset = ff_mnist.FF_CIFAR100(opt, partition, num_classes=100)
    else:
        raise ValueError("Unknown dataset.")

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=3,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_senti_partition(opt, partition):
    # load reviews data
    # print(os.path.join(get_original_cwd(), opt.input.training_path))
    # train_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.training_path), names=["filename", "split", "labels", "features"])
    # test_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.test_path), names=["filename", "split", "labels", "features"])
    # train_df = pd.read_csv('reviews_train.csv',names=["filename", "split", "labels", "features"])
    # test_df = pd.read_csv('reviews_test.csv',names=["filename", "split", "labels", "features"])
    if os.path.exists(os.path.join(get_original_cwd(), opt.input.training_path)) and os.path.exists(os.path.join(get_original_cwd(), opt.input.test_path)): 
        print("Loading preprocessed data")
        train_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.training_path))
        test_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.test_path)) 

    else:
        print("Preprocessing data")
        data_cleaning.DataCleaningPipeline(output_dir = os.path.join(get_original_cwd(), opt.input.path)).process_and_save_data()
        train_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.training_path))
        test_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.test_path))  
        
    
    #train_df = train_df.drop(columns=["filename", "split"])
    #test_df = test_df.drop(columns=["filename", "split"])

    train_labels = torch.tensor(train_df['sentiment'].values, dtype=torch.long)
    test_labels = torch.tensor(test_df['sentiment'].values, dtype=torch.long)

    train_data = train_df['review'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32)) # type: ignore
    train_data = torch.stack([torch.tensor(x) for x in train_data])

    test_data = test_df['review'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32)) # type: ignore
    test_data = torch.stack([torch.tensor(x) for x in test_data])

    final_train_data = torch.hstack((train_data, torch.unsqueeze(train_labels, 1)))
    final_test_data = torch.hstack((test_data, torch.unsqueeze(test_labels, 1)))

    if partition == "train":
        return train_data, train_labels
    else:
        return test_data, test_labels

def get_CIFAR10_partition(opt, partition):
    transform = Compose([
            ToTensor(),
            Lambda(lambda x : (x >= 0.5).float())
            # Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
            ])
    if partition in ["train"]:
        cifar = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        cifar = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return cifar


def get_CIFAR100_partition(opt, partition):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5074, 0.4867, 0.4411],
                                std=[0.2011, 0.1987, 0.2025])])
        
    
    if partition in ["train"]:
        cifar = torchvision.datasets.CIFAR100(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        cifar = torchvision.datasets.CIFAR100(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError(f"Partition {partition} is not implemented")
    

    return cifar

def get_MNIST_partition(opt, partition):

    transform=Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
        ])

    if partition in ["train"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return mnist

def dict_to_cuda(opt, obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = value.to(opt.device, non_blocking=True)
    else:
        obj = obj.to(opt.device, non_blocking=True)
    return obj


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device or "mps" in opt.device:
        inputs = dict_to_cuda(opt, inputs)
        labels = dict_to_cuda(opt, labels)
    return inputs, labels
 
# cools down after the first half of the epochs
def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size
    
def get_prediction_CNN(h , n_classes):
    """
    Generalized prediction function for a model with convolutional layers and a classifier.
    
    Parameters:
    - h: Input tensor of shape [B, C, H, W].
    - n_classes: Number of classes for prediction.
    - sf_pred: Boolean indicating if softmax-based predictions are used (default: False).
    - classifier_b1: Classifier function for softmax predictions (optional).

    Returns:
    - mean squared values: Tensor of shape [B, n_classes] with mean squared values for each class.
    """

    # Step 1: Reshape tensor to group channels by class
    h_reshaped = h.view(h.shape[0], n_classes, h.shape[1] // n_classes, h.shape[2], h.shape[3])

    # Step 2: Compute mean squared value for each group
    mean_squared_values = (h_reshaped ** 2).mean(dim=[2, 3, 4])

    return mean_squared_values


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value
    wandb.log(partition_scalar_outputs, step=epoch)

# create save_model function
def save_model(model):
    torch.save(model.state_dict(), f"{wandb.run.name}-model.pt")
    # log model to wandb
    wandb.save(f"{wandb.run.name}-model.pt")


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict

def overlay_y_on_x3d(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    with torch.no_grad():

        B, C, H, W = x.shape
        unflatten = nn.Unflatten(1, torch.Size([C, H, W]))

        x_ = x.clone()
        x_ = x_.reshape(x_.size(0), -1)

        x_[:, :10] *= 0.0
        x_[range(x_.shape[0]), y] = x_.max()

        x_ = unflatten(x_)


        return x_


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    with torch.no_grad():
        x_ = x.clone()
        x_[:, :10] *= 0.0
        batch_range = range(x.shape[0])
        x_[batch_range, y] = x.max()

        return x_


def overlay_y_on_x4d(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    with torch.no_grad():
        B, C, H, W = x.shape

        unflatten = nn.Unflatten(1, torch.Size([H, W]))

        x_ = x.clone()
        for ch in range(C):
            channel = x_[:, ch]
            # print(channel.shape)
            channel = channel.reshape(channel.size(0), -1)
            channel[:, :10] *= 0.0
            channel[range(channel.shape[0]), y] = channel.max()
            channel = unflatten(channel)
            # print(channel.shape)
            x_[:, ch, :, :] = channel

        return x_


import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_autoencoder_results(inputs, outputs=None, num_images=5, figsize=(15, 15), title="Generated Images"):
    """
    Visualizes input and output images from an autoencoder side-by-side or generated images.
    
    Parameters:
        inputs (torch.Tensor or np.ndarray): The input images, of shape (N, H, W) or (N, C, H, W).
        outputs (torch.Tensor or np.ndarray, optional): The output images, same shape as inputs. Set to None for generated-only visualization.
        num_images (int): Number of images to display in the grid.
        figsize (tuple): Size of the matplotlib figure.
        title (str): Title for generated-only images.
    
    Returns:
        None
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    if outputs is not None and isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    
    # Handle grayscale images (N, H, W) or (N, C, H, W)
    if inputs.ndim == 4:  # (N, C, H, W) - CIFAR-10 (3, 32, 32)
        inputs = np.transpose(inputs, (0, 2, 3, 1))  # Convert to (N, H, W, C)
        if outputs is not None:
            outputs = np.transpose(outputs, (0, 2, 3, 1))  # Convert to (N, H, W, C)
    
    # Clip values for visualization if needed
    inputs = np.clip(inputs, 0, 1)  # Assuming normalized inputs
    if outputs is not None:
        outputs = np.clip(outputs, 0, 1)  # Assuming normalized outputs

    # Select random indices for visualization
    indices = np.random.choice(inputs.shape[0], num_images, replace=False)

    # If outputs are provided, compare inputs and outputs side-by-side
    if outputs is not None:
        fig, axes = plt.subplots(2, num_images, figsize=figsize)
        for i, idx in enumerate(indices):
            # Input images
            axes[0, i].imshow(inputs[idx])  # Automatically handles RGB
            axes[0, i].axis('off')
            axes[0, i].set_title("Input")

            # Output images
            axes[1, i].imshow(outputs[idx])  # Automatically handles RGB
            axes[1, i].axis('off')
            axes[1, i].set_title("Output")

    # If outputs are not provided, visualize generated images only
    else:
        fig, axes = plt.subplots(1, num_images, figsize=figsize)
        for i, idx in enumerate(indices):
            axes[i].imshow(inputs[idx])  # Automatically handles RGB
            axes[i].axis('off')
            axes[i].set_title(title)
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()

def generate_images(decoder, y, y_neg, device,mode='uniform', num=400, grid_size=0.05):
    """
    Generates MNIST imgaes with 2D latent variables sampled uniformly with mean 0

    Args:
        mode: 'uniform' or 'random'
        num: Number of samples to make. Accepts square numbers
        grid_size: Distance between adjacent latent variables
        PATH: The path to saved model (saved with torch.save(model, path))
        model: The trained model itself
    
    Note:
        Specify only one of PATH or model, not both
    """

    # Check arguments
    if mode!='uniform' and mode!='random':
        raise ValueError("Argument mode should either be 'uniform' or 'random'")
    if num!=(int(num**0.5))**2:
        raise ValueError('Argument num should be a square number')
    
    # Sample tensor of latent variables
    if mode == 'uniform':
        side = num**0.5
        axis = (torch.arange(side) - side//2) * grid_size
        x = axis.reshape(1, -1)
        y = x.transpose(0, 1)
        z = torch.stack(torch.broadcast_tensors(x, y), 2).reshape(-1, 2).to(device)
    elif mode == 'random':
        z = torch.randn((num, 2), device=device) # nx2

    # Generate output from decoder
    with torch.no_grad():
        for i in range(10):
            label = torch.zeros((num, 10), device=device)
            label[:, i] = 1
            label_n = label[torch.randperm(label.size(0))]
            latent = torch.cat((z, label), dim=1)
            latent_n = torch.cat((z, label_n), dim=1)
            with torch.no_grad():
                output = decoder(latent, latent_n, y, y_neg ,mode = "test")
            display_and_save_batch(f'{mode}-generation', output)
    

def plot_latent_space(decoder, device, scale=1.0, n=10, image_size=32, channels=3, figsize=15):
    """
    Visualize the 2D latent space of a decoder model by generating images over a grid.
    
    Parameters:
        model: The trained model with a `decode` method.
        scale (float): The range of the latent space to sample.
        n (int): Number of grid points along each axis.
        image_size (int): The size of the output images (e.g., 32 for CIFAR-10).
        channels (int): Number of channels in the output images (3 for RGB).
        figsize (float): Size of the matplotlib figure.
    """
    # Create a blank canvas to store the grid of generated images
    figure = np.zeros((image_size * n, image_size * n, channels))
    
    # Define the grid of latent space points
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]  # Reverse for correct orientation
    
    # Loop over each grid point and generate an image
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # Create a latent space sample
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            
            # Decode the sample into an image
            # Pass the latent vectors through the decoder
            h_pos = decoder(z_sample)
            image = h_pos[0].detach().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            image = np.clip(image, 0, 1)  # Ensure valid pixel values
            
            # Place the image in the canvas
            figure[
                i * image_size : (i + 1) * image_size,
                j * image_size : (j + 1) * image_size,
            ] = image
    
    # Plot the figure
    plt.figure(figsize=(figsize, figsize))
    plt.title("VAE Latent Space Visualization")
    plt.xticks(
        np.arange(image_size // 2, image_size * n, image_size),
        labels=np.round(grid_x, 1),
    )
    plt.yticks(
        np.arange(image_size // 2, image_size * n, image_size),
        labels=np.round(grid_y, 1),
    )
    plt.imshow(figure, extent=(0, n, 0, n), origin="upper")
    plt.axis("off")
    plt.show()


def display_and_save_batch(title, batch, save=True, display=True):
    """
    Display and save a batch of images using matplotlib.
    Parameters:
        - title: Title of the plot and filename for saving.
        - batch: Tensor of images (B, C, H, W).
        - save: Boolean to save the image.
        - display: Boolean to display the image.
    """
    # Convert the batch to a grid
    im = torchvision.utils.make_grid(batch, nrow=int(batch.shape[0]**0.5))
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(np.transpose(im.cpu().numpy(), (1, 2, 0)), cmap='gray')
    if save:
        os.makedirs("results", exist_ok=True)
        plt.savefig(f'results/{title}.png', transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()