import math

import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms

from src_other import Layer_cnn, Layer_fc
from src import utils


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers
        self.layer_norm = LayerNorm()

        if self.opt.device == "mps":
            import os
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        
        input_layer_size = utils.get_input_layer_size(opt)

        # Initialize the model.
        print("Model structure: ", self.opt.model.structure== "CwC")
        if self.opt.model.structure == "CwC":
            # For CwC model
            self.iter = 1

            self.batch_size = self.opt.input.batch_size
            self.show_iters = 800
            self.sf_pred = self.opt.CwC.sf_pred
            self.channel_list = self.opt.CwC.channel_list
            self.final_channels = self.channel_list[-1]
            self.ClassGroups = False
            self.n_classes = self._N_classes()
            self.cfse = self.opt.CwC.CFSE
            self.dataset = self.opt.input.dataset
            self.ilt = self.opt.CwC.ILT
            self.loss_CwC = self.opt.CwC.loss


            self.nn_layers = []
            
            self.model = nn.ModuleList()

            


            if self.opt.input.dataset == 'mnist':
                if self.ilt == 'mnist':
                    self.start_end = [[0, 3], [1, 4], [2, 5], [3, 6], [4, 20], [5, 20]]
                else:
                    # self.start_end = [[0, 6], [0, 11], [0, 16], [0, 21], [0, 20], [0, 20]]
                    self.start_end = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 20], [0, 20]]
                CNN_l1_dims = [1, 28, 28]
            elif self.opt.input.dataset == 'fmnist':
                if self.ilt == 'Fast':
                    self.start_end = [[0, 7], [1, 10], [2, 13], [3, 16], [4, 30], [5, 40]]
                else:
                    self.start_end = [[0, 10], [0, 15], [0, 19], [0, 23], [0, 36], [0, 50]]
                    # self.start_end = [[0, 6], [0, 9], [0, 11], [0, 14], [0, 30], [0, 40]]
                CNN_l1_dims = [1, 28, 28]
            else:
                if self.ilt == 'Fast':
                    self.start_end = [[0, 11], [2, 18], [4, 26], [6, 32], [8, 36], [10, 50]]
                else:
                    self.start_end = [[0, 11], [0, 16], [0, 21], [0, 25], [0, 36], [0, 50]]
                    # self.start_end = [[0, 30], [0, 30], [30, 60], [30, 60], [60, 85], [60, 85], [85, 100], [85, 100]]
                CNN_l1_dims = [3, 32, 32]

            # # # CNN LAYERS # # #
            kernel_size = 3

            # Dynamically add layers
            dims = [CNN_l1_dims]

            
            for i, out_channels in enumerate(self.channel_list):
                if self.ClassGroups:
                    #ClassGroup case for CIFAR-100
                    if i % 2 == 1 and self.cfse:
                        group = self.n_classes[i]
                    else:
                        group = 1

                    if max(self.n_classes) == self.n_classes[i]:
                        class_groups = None
                    else:
                        class_groups = int(max(self.n_classes)/self.n_classes[i])
                    
                    in_channels = dims[-1][0] # [[1, 28, 28]]
                    layer = Layer_cnn.Conv_Layer(dims[-1], opt= self.opt, in_channels=in_channels, out_channels=out_channels, num_classes=self.n_classes[i],
                                    kernel_size=kernel_size, maxpool=(i % 2 == 1), groups=group, droprate=0, loss_criterion=self.loss_CwC, ClassGroups=  class_groups).to(self.opt.device)
                    self.model.append(layer)
                    dims.append(layer.next_dims)
                else:
                    # deactivate class groups
                    class_groups = None
                    # if CSFE is activated, the group is the number of classes
                    if i % 2 == 1 and self.cfse:
                        group = self.n_classes
                    else:
                        group = 1
                    
                    in_channels = dims[-1][0] # [[1, 28, 28]]
                    layer = Layer_cnn.Conv_Layer(dims[-1], opt= self.opt, in_channels=in_channels, out_channels=out_channels, num_classes=self.n_classes,
                                    kernel_size=kernel_size, maxpool=(i % 2 == 1), groups=group, droprate=0, loss_criterion=self.loss_CwC, ClassGroups=  class_groups).to(self.opt.device)
                    self.model.append(layer)
                    dims.append(layer.next_dims)

        elif self.opt.model.structure == "AE":
            
            self.batch_size = self.opt.input.batch_size
            self.show_iters = 800
            self.sf_pred = self.opt.AE.sf_pred
            self.channel_list = self.opt.AE.channel_list
            self.dataset = self.opt.input.dataset
            self.loss_AE = self.opt.AE.loss
            self.n_classes = self._N_classes()
            self.cfse = self.opt.AE.CFSE
            self.act_fn_AE = "leakyrelu"
            self.batchnorm = True


            self.enc_model = nn.ModuleList()
            self.dec_model = nn.ModuleList()

            if self.opt.input.dataset == 'mnist':
                CNN_l1_dims = [1, 28, 28]
            elif self.opt.input.dataset == 'fmnist':
                CNN_l1_dims = [1, 28, 28]
            else:
                CNN_l1_dims = [3, 32, 32]

            # # # CNN LAYERS # # #
            kernel_size = 3

            # Dynamically add layers
            dims = [CNN_l1_dims]

            for i, out_channels in enumerate(self.channel_list):
                self.kernel = self.opt.AE.enc_kernel[i]
                 # deactivate class groups
                class_groups = None
                # if CSFE is activated, the group is the number of classes
                if i % 2 == 1 and self.cfse:
                    group = self.n_classes
                else:
                    group = 1
                

                in_channels = dims[-1][0] # [[1, 28, 28]]
                layer = Layer_cnn.Conv_Layer(dims[-1], opt= self.opt, in_channels=in_channels, out_channels=out_channels, num_classes=self.n_classes,
                                kernel_size=self.kernel["kernel_size"], stride = self.kernel["stride"], padding=self.kernel["padding"], maxpool=False,
                                act_fn = self.act_fn_AE, groups=group, droprate=0, loss_criterion=self.loss_AE, ClassGroups = class_groups).to(self.opt.device)
                self.enc_model.append(layer)
                dims.append(layer.next_dims)

            self.channel_list = list(reversed(self.channel_list[:-1]))  # Convert back to a list
            self.channel_list.append(CNN_l1_dims[0])  # Append now works
            dims = [self.enc_model[-1].next_dims]

            for i, out_channels in enumerate(self.channel_list):
                self.kernel = self.opt.AE.dec_kernel[i]
                 # deactivate class groups
                class_groups = None
                # if CSFE is activated, the group is the number of classes
                if i % 2 == 1 and self.cfse:
                    group = self.n_classes
                else:
                    group = 1
                
                # Switch from kernel latent to normal kernel
                if i==len(self.channel_list)-1:
                    self.batchnorm = False
                    self.act_fn_AE = "tanh"
                
                in_channels = dims[-1][0] # [[1, 28, 28]]
                layer = Layer_cnn.Conv_Layer_transpose(dims[-1], opt= self.opt, in_channels=in_channels, out_channels=out_channels, num_classes=self.n_classes,
                                kernel_size=self.kernel["kernel_size"], stride = self.kernel["stride"], padding=self.kernel["padding"], maxpool=False,
                                batchnorm=self.batchnorm, act_fn = self.act_fn_AE, groups=group, droprate=0, loss_criterion=self.loss_AE, ClassGroups=  class_groups).to(self.opt.device)
                self.dec_model.append(layer)
                dims.append(layer.next_dims)
            
        elif self.opt.model.structure == "VAE" or self.opt.model.structure == "FFCVAE":
            self.batch_size = self.opt.input.batch_size
            self.show_iters = 800
            self.sf_pred = self.opt.VAE.sf_pred
            self.enc_channel_list = self.opt.VAE.enc_channel_list
            self.dec_channel_list = self.opt.VAE.dec_channel_list
            self.dataset = self.opt.input.dataset
            self.loss_AE = self.opt.VAE.loss
            self.n_classes = self._N_classes()
            self.cfse = self.opt.VAE.CFSE
            self.maxpool = False
            self.beta = self.opt.VAE.beta
            self.C_max = torch.Tensor([self.opt.VAE.max_capacity])
            self.latent_dim  = self.opt.VAE.latent_dim
            self.latent_FF = self.opt.VAE.latent_FF
            self.latent_shape = self.opt.VAE.latent_shape
            self.batchnorm_dec = self.opt.VAE.batchnorm_dec
            self.batchnorm_enc = self.opt.VAE.batchnorm_enc
            self.relu_dec = self.opt.VAE.relu_dec
            self.relu_enc = self.opt.VAE.relu_enc

            self.enc_model = nn.ModuleList()
            self.dec_model = nn.ModuleList()

            if self.opt.input.dataset == 'mnist':
                CNN_l1_dims = [1, 28, 28]
            elif self.opt.input.dataset == 'fmnist':
                CNN_l1_dims = [1, 28, 28]
            else:
                CNN_l1_dims = [3, 32, 32]

            # # # CNN LAYERS # # #
            kernel_size = 3

            # Dynamically add layers
            dims = [CNN_l1_dims]
            self.image_size = CNN_l1_dims[1]

            for i, out_channels in enumerate(self.enc_channel_list):
                self.kernel = self.opt.VAE.enc_kernel[i]
                 # deactivate class groups
                class_groups = None
                # if CSFE is activated, the group is the number of classes
                if i % 2 == 1 and self.cfse:
                    group = self.n_classes
                else:
                    group = 1

                
                

                in_channels = dims[-1][0] # [[1, 28, 28]]
                layer = Layer_cnn.Conv_Layer(dims[-1], opt= self.opt, in_channels=in_channels, out_channels=out_channels, num_classes=self.n_classes, act_fn = self.relu_enc[i],
                                kernel_size=self.kernel["kernel_size"], stride = self.kernel["stride"], padding=self.kernel["padding"], maxpool=self.maxpool, batchnorm=self.batchnorm_enc[i],
                                groups=group, droprate=0, loss_criterion=self.loss_AE, ClassGroups = class_groups).to(self.opt.device)
                self.enc_model.append(layer)
                dims.append(layer.next_dims)
                self.maxpool = False

            # Layer for latent space
            self.fc = Layer_fc.FC_LayerCW(self.enc_channel_list[-1]*self.latent_shape[0]* self.latent_shape[1], 1024, relu = True, dropout = False, normalize =False, batchnorm =True).to(self.opt.device)
            # self.fc_mu = nn.Linear(self.enc_channel_list[-1]*4, self.latent_dim)
            self.fc_mu_var = Layer_fc.FC_LayerCW(1024, 2*self.latent_dim, relu = False, dropout = False, normalize =False, batchnorm =False).to(self.opt.device)

            self.decoder_input_0 = Layer_fc.FC_LayerCW(self.latent_dim+10, 1024, relu = True, dropout = False, normalize =False, batchnorm =True).to(self.opt.device)
            self.decoder_input_1 = Layer_fc.FC_LayerCW(1024, self.enc_channel_list[-1]*self.latent_shape[0]* self.latent_shape[0], relu = True, dropout = False, normalize =False, batchnorm =True).to(self.opt.device)
            # self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)


            
            dims = [self.enc_model[-1].next_dims]
            

            for i, out_channels in enumerate(self.dec_channel_list):
                self.kernel = self.opt.VAE.dec_kernel[i]
                 # deactivate class groups
                class_groups = None
                # if CSFE is activated, the group is the number of classes
                if i % 2 == 1 and self.cfse:
                    group = self.n_classes
                else:
                    group = 1
                
                
            
                in_channels = dims[-1][0] # [[1, 28, 28]]
                layer = Layer_cnn.Conv_Layer_transpose(dims[-1], opt= self.opt, in_channels=in_channels, out_channels=out_channels, num_classes=self.n_classes, act_fn=self.relu_dec[i],
                                kernel_size=self.kernel["kernel_size"], stride = self.kernel["stride"], padding=self.kernel["padding"], output_padding=self.kernel["output_padding"] , maxpool=False,
                                batchnorm=self.batchnorm_dec[i],  groups=group, droprate=0, loss_criterion=self.loss_AE, ClassGroups=  class_groups).to(self.opt.device)
                self.dec_model.append(layer)
                dims.append(layer.next_dims)

            layers = []
            layers.append(nn.Linear(self.latent_dim + self.n_classes, self.n_classes))

            if self.opt.model.maxsubtract:
                layers.append(MaxSubtractLayer())
            if self.opt.model.softmax:
                layers.append(nn.Softmax(dim=1))
            
            self.linear_classifier = nn.Sequential(*layers)

        elif self.opt.model.structure == "VFFAE":
            self.n_classes = self._N_classes()

            if self.opt.input.dataset == 'mnist':
                CNN_l1_dims = [1, 28, 28]
                self.image_size = CNN_l1_dims[1]

                enc_dims = [self.image_size**2, 500, 250]
                latent_dim = 10
                dec_dims = [250, 500, self.image_size**2]
            elif self.opt.input.dataset == 'fmnist':
                CNN_l1_dims = [1, 28, 28]
            else:
                CNN_l1_dims = [3, 32, 32]

            
            if self.opt.input.dataset == "mnist":
                self.enc_model = nn.ModuleList([
                    Layer_fc.FC_LayerVFFAE_enc(enc_dims[0], enc_dims[1], act_fn="relu", dropout=False, normalize=False, layernorm=True, batchnorm=False).to(self.opt.device),
                    Layer_fc.FC_LayerVFFAE_enc(enc_dims[1], enc_dims[2], act_fn="relu", dropout=False, normalize=False, layernorm=True, batchnorm=False).to(self.opt.device),
                    Layer_fc.FC_LayerVFFAE_enc(enc_dims[2], latent_dim, act_fn="relu", dropout=False, normalize=False, layernorm=False, batchnorm=False).to(self.opt.device)
                ])

                # Decoder Layers (using ModuleList)
                self.dec_model = nn.ModuleList([
                    Layer_fc.FC_LayerVFFAE_dec(latent_dim, dec_dims[0], act_fn="relu", dropout=False, normalize=False, layernorm=True, batchnorm=False).to(self.opt.device),
                    Layer_fc.FC_LayerVFFAE_dec(dec_dims[0], dec_dims[1], act_fn="relu", dropout=False, normalize=False, layernorm=True, batchnorm=False).to(self.opt.device),
                    Layer_fc.FC_LayerVFFAE_dec(dec_dims[1], dec_dims[2], act_fn="sigmoid", dropout=False, normalize=False, layernorm=True, batchnorm=False).to(self.opt.device)
                ])

        else:
            # Initialize forward-forward loss.
            self.ff_loss = nn.BCEWithLogitsLoss()
            self.n_classes = self._N_classes()
            self.act_fn = ReLUFullGrad()

            # Initialize peer normalization loss.
            self.running_means = [
                torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
                for i in range(self.opt.model.num_layers)
            ]# [784,2000,2000,2000]


            self.model = nn.ModuleList([(nn.Linear(input_layer_size, self.num_channels[0]))]
                                                    )
            for i in range(1, len(self.num_channels)):
                self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i])
                                  )

        


        
        # Initialize downstream classification loss.
        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.model.num_layers - 1)
        ) # 2000+2000+2000 = 6000 
        
        # Create the linear classifier for all architectures
        self.droprate = self.opt.model.droprate
        layers = []

        # Conditionally add Dropout if self.opt.model.dropout is True
        if self.opt.model.dropout:
            layers.append(torch.nn.Dropout(p=self.droprate))

        # Add the Linear layer
        if self.opt.model.structure == "CwC" and self.ClassGroups:
            layers.append(nn.Linear(self.model[-1].N_neurons_out, self.n_classes[-1], bias=False))
        elif self.opt.model.structure == "CwC" and not self.ClassGroups:
            layers.append(nn.Linear(self.model[-1].N_neurons_out, self.n_classes, bias=False))
        if self.opt.VFFAE.classifier and self.opt.model.structure == "VFFAE":
                layers.append(nn.Linear(latent_dim, self.n_classes))
        else:
            layers_multi_pass = layers.copy()
            layers_pass = layers.copy()
            layers_multi_pass.append(nn.Linear(channels_for_classification_loss, self.n_classes, bias=False))
            layers_pass.append(nn.Linear(self.opt.model.hidden_dim, self.n_classes, bias=False))
            
        
            

        # Add the MaxSubtractLayer and Softmax
        if self.opt.model.structure == "CwC" or self.opt.model.structure == "VFFAE":
            if self.opt.model.maxsubtract:
                layers.append(MaxSubtractLayer())
            if self.opt.model.softmax:
                layers.append(nn.Softmax(dim=1))
            
            self.linear_classifier = nn.Sequential(*layers)
        else: 
            if self.opt.model.maxsubtract:
                layers_pass.append(MaxSubtractLayer())
            if self.opt.model.softmax:
                layers_pass.append(nn.Softmax(dim=1))

            if self.opt.model.maxsubtract:
                layers_multi_pass.append(MaxSubtractLayer())
            if self.opt.model.softmax:
                layers_multi_pass.append(nn.Softmax(dim=1))
            # Create the sequential model
            self.linear_classifier_pass = nn.Sequential(*layers_pass)
            self.linear_classifier_multi_pass = nn.Sequential(*layers_multi_pass)
        

        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

        

    def _init_weights(self):
        if self.opt.model.structure == "AE" or self.opt.model.structure == "VAE" or self.opt.model.structure == "VFFAE" or self.opt.model.structure == "FFCVAE":
            for m in self.enc_model.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            for m in self.dec_model.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
        else:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(
                        m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                    )
                    torch.nn.init.zeros_(m.bias)

        if self.opt.model.structure == "CwC" or self.opt.model.structure == "VFFAE" or self.opt.model.structure == "FFCVAE":
            for m in self.linear_classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
        else:
            for m in self.linear_classifier_pass.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)

            for m in self.linear_classifier_multi_pass.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)

    def _N_classes(self):
        if self.opt.input.dataset == "cifar10" or self.opt.input.dataset == "mnist" or self.opt.input.dataset == "senti":
            return 10
        elif self.opt.input.dataset == "cifar100":
            if self.opt.model.structure != "CwC":
                return 100
            if not hasattr(self.opt.CwC, "N_Classes"):
                raise ValueError(
                    "Error: CIFAR-10 dataset is selected, but no class group division is provided. "
                    "Please specify 'ClassGroup' in the dataset configuration."
                )
            else:
                self.ClassGroups = True
                return self.opt.CwC.N_Classes

    def _calc_sparsity(self, z):
        return None


    # loss incentivizing the mean activity of neurons in a layer to have low variance
    def _calc_peer_normalization_loss(self, idx, z): # z is bs*2, 2000
        # Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0) #bsx2000 -> 2000

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        ) # the detach means that the gradient because of previous batches is not backpropagated. only the current mean activity is backpropagated
        # running_mean * 0.9 + mean_activity * 0.1

        # 2000
        # 1 = mean activation across entire layer
        # 

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)  # Sum of squares of each activation, shape: (batch_size,)
        
        threshold = self.opt.model.threshold
        logits = sum_of_squares - z.shape[1] * threshold  # Compute logits based on the threshold
        ff_loss = self.ff_loss(logits, labels.float())  # Calculate loss

        # Calculate accuracy and identify correctly classified images
        with torch.no_grad():
            # Determine which images are classified correctly
            predictions = torch.sigmoid(logits) > 0.5
            correct_classifications = predictions == labels  # Boolean tensor of correctly classified samples

            # Calculate accuracy
            ff_accuracy = torch.sum(correct_classifications).item() / z.shape[0]

            # Get the indices or the images themselves that were classified correctly
            ff_sparsity = (torch.sqrt(torch.tensor(z.shape[1])) - (torch.sum(torch.abs(z[correct_classifications]), dim=-1)/torch.sum(z[correct_classifications] ** 2, dim=-1))) / (torch.sqrt(torch.tensor(z.shape[1])) - 1)  # Selects only the correctly classified samples

            # Average over all sparsity values to get the sparsity of the true classified images
            ff_sparsity = torch.mean(ff_sparsity)

        return ff_loss, ff_accuracy, ff_sparsity

    def elbo_loss(self, recon_x, x, mu, log_var):
        """
        ELBO Optimization objective for gaussian posterior
        (reconstruction term + regularization term)
        """
        # MSE
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # CROSS ENTROPY
        # x = x * 255
        # x.data = x.data.int().long().view(-1)
        # # print(recon_x.shape)
        # recon_x = recon_x.permute(0, 2, 3, 4, 1)  # N * C * W * H
        # # print(recon_x.shape)
        # recon_x = recon_x.contiguous().view(-1, 256)
        # recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')


        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Or this
        # kld_loss = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var) 
        # kld_loss = torch.sum(kld_loss).mul_(-0.5)

        # Return the combined loss (reconstruction + regularization)
        # print(f"MSE: {MSE}, KLD: {kld_loss}")
        return recon_loss, kld_loss
        # return -torch.mean(elbo)
    

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1)

        Params:
            mu (Tensor): Mean of Gaussian latent variables [B x D]
            logvar (Tensor): log-Variance of Gaussian latent variables [B x D]

        Returns: 
            z (Tensor) [B x D]
        """

        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = eps.mul(sigma).add_(mu)

        return z
    
   
    def forward(self, inputs, labels, epoch):
        scalar_outputs = {"Loss": torch.zeros(1, device=self.opt.device)}
        if self.opt.model.structure != "CwC" and self.opt.model.structure != "BP" and self.opt.model.structure != "AE" or self.opt.model.structure != "VAE":
            scalar_outputs["Peer Normalization"] = torch.zeros(1, device=self.opt.device)

        # Dictionary mapping model structure to methods
        structure_methods = {
            "FF": self._forward,
            "BP": self._forward,
            "BP/FF": self._forward,
            "ParBP": self._forward,
            "CwC": self._forward_CwC,
            "AE": self._forward_AE,
            "VAE": self._forward_VAE,
            "VFFAE": self._forward_VFFAE,
            "FFCVAE": self._forward_FFCVAE
        }
        
        # Call the method based on the model structure
        forward_method = structure_methods.get(self.opt.model.structure)
        
        if self.opt.model.structure == "CwC" and forward_method:
            scalar_outputs = forward_method(inputs, labels, scalar_outputs, epoch)
        elif forward_method:
            scalar_outputs = forward_method(inputs, labels, scalar_outputs)
        else:
            raise ValueError(f"Unsupported model structure: {self.opt.model.structure}")

        return scalar_outputs

    def _forward(self, inputs, labels, scalar_outputs):
        # print(inputs["pos_images"].shape) # bs, 1, 28, 28
        # print(inputs["neg_images"].shape) # bs, 1, 28, 28
        # print(inputs["neutral_sample"].shape) # bs, 1, 28, 28
        # print(labels["class_labels"].shape) # bs
        # exit()
        # Concatenate positive and negative samples and create corresponding labels.
        if self.opt.model.structure == "FF" or  self.opt.model.structure=="BP/FF" or self.opt.model.structure=="ParBP":
            z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0) # 2*bs, 1, 28, 28
            posneg_labels = torch.zeros(z.shape[0], device=self.opt.device) # 2*bs
            posneg_labels[: self.opt.input.batch_size] = 1 # first BS samples true, next BS samples false
        elif self.opt.model.structure == "BP":
            z = inputs["neutral_sample"]

        z = z.reshape(z.shape[0], -1) # 2*bs, 784
        z = self.layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn(z)

            if self.opt.model.structure == "FF" or self.opt.model.structure == "BP/FF" or self.opt.model.structure == "ParBP":

                if self.opt.model.peer_normalization > 0:
                    peer_loss = self._calc_peer_normalization_loss(idx, z)
                    scalar_outputs["Peer Normalization"] += peer_loss
                    scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

                ff_loss, ff_accuracy, ff_sparsity = self._calc_ff_loss(z, posneg_labels)
                scalar_outputs[f"loss_layer_{idx}"] = ff_loss
                scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
                scalar_outputs[f"ff_sparsity_layer_{idx}"] = ff_sparsity
                scalar_outputs["Loss"] += ff_loss

            if self.opt.model.structure == "FF":
                z = z.detach()

            if self.opt.model.structure == "ParBP" and (idx+1)%self.opt.model.parbp_steps != 0:
                z = z.detach()

            z = self.layer_norm(z)
        
        
        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )


        if self.opt.model.structure == "FF":
            scalar_outputs = self.forward_downstream_multi_pass(
                inputs, labels, scalar_outputs=scalar_outputs
            )

        return scalar_outputs
        

    def _forward_AE(self, inputs, labels, scalar_outputs):
        # print(f"inputs:{inputs} \n labels:{labels}")
        z = inputs
        y = labels.long()
        y_n = y[torch.randperm(z.size(0))]

        h_pos = z

        for i, convlayer in enumerate(self.enc_model):
            h_pos, ff_loss = convlayer.forward_forward(h_pos, y)
            # print("Shape after layer",i, "encoder", h_pos.shape)
            h_pos.detach()
            scalar_outputs["Loss"] += ff_loss

        for i, convlayer in enumerate(self.dec_model):
            if i == len(self.dec_model) - 1:
                h_pos = convlayer.forward(h_pos)
                # print("Shape after layer",i, "decoder", h_pos.shape)
            else:
                h_pos, ff_loss = convlayer.forward_forward(h_pos, y)
                # print("Shape after layer",i, "decoder", h_pos.shape)
                scalar_outputs["Loss"] += ff_loss
                h_pos.detach()
        
        loss = F.mse_loss(h_pos, inputs)

        scalar_outputs["Loss"] += loss
        scalar_outputs["MSE_loss"] = loss

        return scalar_outputs
    
    def _encoder_CVAE(self, inputs, labels, scalar_outputs, mode = "train"):
        z = inputs
        y = labels.long()
        y_n = y[torch.randperm(z.size(0))]

        y = y.view(z.shape[0], 1)
        y_n = y_n.view(z.shape[0], 1)
        # positive
        onehot_y = torch.zeros((z.shape[0], 10), requires_grad=False, device= self.opt.device)
        onehot_y.scatter_(1, y, 1)
        # negative
        onehot_y_neg = torch.zeros((z.shape[0], 10), requires_grad=False, device= self.opt.device)
        onehot_y_neg.scatter_(1, y_n, 1)

        onehot_conv_y = onehot_y.view(z.shape[0], 1, 1, 10)*torch.ones((z.shape[0], z.shape[1], z.shape[2], 10), device= self.opt.device)
        h_pos = torch.cat((z, onehot_conv_y), dim=3)

        if mode == "train":
            for i, convlayer in enumerate(self.enc_model):
                h_pos, ff_loss = convlayer.forward_forward(h_pos, y)
                # print("Shape after layer",i, "encoder", h_pos.shape
                scalar_outputs["Loss"] += ff_loss
                h_pos.detach()
        elif mode== "test":
            for i, convlayer in enumerate(self.enc_model):
                h_pos = convlayer.forward(h_pos)
                # print("Shape after layer",i, "encoder", h_pos.shape)

        return h_pos, y, y_n, onehot_y, onehot_y_neg, scalar_outputs
    
    def _latent_CVAE(self, h_pos, y, y_n, onehot_y, onehot_y_neg, scalar_outputs):
        
        if self.latent_FF:
            h_pos = utils.overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
            h_neg = utils.overlay_y_on_x3d(h_pos, y_n)

            # Flatten
            # print("Shape before latent space", h_pos.shape)
            h_pos, h_neg, nn_loss = self.fc.forward_forward(h_pos, h_neg, y)
            scalar_outputs["Loss"] += nn_loss
            h_pos.detach()
            h_neg.detach()

            mu_var = self.fc_mu_var.forward(h_pos)
            # split x in half
            mu = mu_var[:, :self.latent_dim]
            # sigma shouldn't be negative
            log_var = mu_var[:, self.latent_dim:]

            h_pos = self.reparameterize(mu, log_var)
            h_neg = torch.cat((h_pos, onehot_y_neg), dim=1)
            h_pos = torch.cat((h_pos,  ), dim=1)
            # print("Shape after latent space", h_pos.shape)

            utils.overlay_y_on_x(h_pos, y)  
            utils.overlay_y_on_x(h_neg, y_n)

            return h_pos, h_neg, mu, log_var, scalar_outputs
        
        else:
            # Flatten
            # print("Shape before latent space", h_pos.shape)
            h_pos = self.fc.forward(h_pos)
            mu_var = self.fc_mu_var.forward(h_pos)
            # split x in half
            mu = mu_var[:, :self.latent_dim]
            # sigma shouldn't be negative
            log_var = mu_var[:, self.latent_dim:]

            h_pos = self.reparameterize(mu, log_var)
            h_pos = torch.cat((h_pos, onehot_y_neg), dim=1)
            # print("Shape after latent space", h_pos.shape)

            return h_pos, mu, log_var, scalar_outputs
        

    def _decoder_CVAE(self, h_pos, y, y_n, h_neg=None, scalar_outputs=None, mode = "train"):
        
        if self.latent_FF:
            if mode == "train":
                h_pos, h_neg, nn_loss= self.decoder_input_0.forward_forward(h_pos, h_neg, y)
                scalar_outputs["Loss"] += nn_loss
                h_pos.detach()
                h_neg.detach()

                utils.overlay_y_on_x(h_pos, y)
                utils.overlay_y_on_x(h_neg, y_n)

                h_pos, h_neg, nn_loss = self.decoder_input_1.forward_forward(h_pos, h_neg, y)
                scalar_outputs["Loss"] += nn_loss
                h_pos.detach()
                h_neg.detach()

                h_pos = h_pos.view(-1, self.opt.VAE.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
            elif mode == "test":
                h_pos = self.decoder_input_0.forward(h_pos)
                h_pos = self.decoder_input_1.forward(h_pos)
                h_pos = h_pos.view(-1, self.opt.VAE.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
        else:
            h_pos= self.decoder_input_0.forward(h_pos)
            h_pos = self.decoder_input_1.forward(h_pos)
            h_pos = h_pos.view(-1, self.opt.VAE.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
            
        if mode == "train":
            for i, convlayer in enumerate(self.dec_model):
                if i == len(self.dec_model) - 1:
                    h_pos = convlayer.forward(h_pos)
                    # print("Shape after layer",i, "decoder", h_pos.shape)
                else:
                    h_pos, ff_loss = convlayer.forward_forward(h_pos, y)
                    # print("Shape after layer",i, "decoder", h_pos.shape)
                    scalar_outputs["Loss"] += ff_loss
                    h_pos.detach()
        elif mode== "test":
            for i, convlayer in enumerate(self.dec_model):
                h_pos = convlayer.forward(h_pos)
                # print("Shape after layer",i, "decoder", h_pos.shape)
        
        if scalar_outputs is None:
            return h_pos
        else:
            return h_pos, scalar_outputs
        


    def _forward_VAE(self, inputs, labels, scalar_outputs):
        # print(f"inputs:{inputs} \n labels:{labels}")
        torch.autograd.set_detect_anomaly(True)
        
        z = inputs

        h_pos, y, y_n, onehot_y, onehot_y_neg, scalar_outputs = self._encoder_CVAE(z, labels, scalar_outputs)
        
        if self.latent_FF:
            h_pos, h_neg, mu, log_var, scalar_outputs = self._latent_CVAE(h_pos, y, y_n, onehot_y, onehot_y_neg, scalar_outputs)
        else:
            
            h_pos, mu, log_var, scalar_outputs = self._latent_CVAE(h_pos, y, y_n, onehot_y, onehot_y_neg, scalar_outputs)
            h_neg = None

        h_pos, scalar_outputs = self._decoder_CVAE(h_pos, y, y_n, h_neg, scalar_outputs)
  
        
        rec_loss, kld_loss = self.elbo_loss(h_pos, inputs, mu, log_var)
        scalar_outputs["Loss"] += rec_loss + kld_loss* self.beta
        scalar_outputs["MSE_loss"] = rec_loss
        scalar_outputs["KLD_loss"] = kld_loss

        
        return scalar_outputs
    
    ### CVAE MIRROR DECODER ###
    def _encoder_CVAE_mirror(self, inputs, labels, scalar_outputs, mode = "train", label = None):
        z = inputs
                
        if mode == "train":
            y = labels.long()
            y_n = y[torch.randperm(z.size(0))]

            enc_out = []
            enc_out.append(z)

            y = y.view(z.shape[0], 1)
            y_n = y_n.view(z.shape[0], 1)
            # positive
            onehot_y = torch.zeros((z.shape[0], 10), requires_grad=False, device= self.opt.device)
            onehot_y.scatter_(1, y, 1)
            # negative
            onehot_y_neg = torch.zeros((z.shape[0], 10), requires_grad=False, device= self.opt.device)
            onehot_y_neg.scatter_(1, y_n, 1)

            onehot_conv_y = onehot_y.view(z.shape[0], 1, 1, 10)*torch.ones((z.shape[0], z.shape[1], z.shape[2], 10), device= self.opt.device)
            h_pos = torch.cat((z, onehot_conv_y), dim=3)
            
            for i, convlayer in enumerate(self.enc_model):

                # print("Shape before layer",i, "encoder", h_pos.shape)
                # print("Shape before layer",i, "encoder", y.shape)
                h_pos, ff_loss = convlayer.forward_forward(h_pos, y)
                # print("Shape after layer",i, "encoder", h_pos.shape
                scalar_outputs["Loss"] += ff_loss
                h_pos.detach()
                enc_out.append(h_pos[:,:,:,:h_pos.shape[2]].clone())
            
            enc_out.reverse()

            return h_pos, y, y_n, onehot_y, onehot_y_neg, enc_out, scalar_outputs

        elif mode== "test":
            activations = []

            if label is not None:
                # Ensure the label is valid
                if not (0 <= label < self.n_classes):
                    raise ValueError(f"Invalid label: {label}. Must be between 0 and {self.n_classes - 1}.")
                
                # Generate one-hot for the selected label
                onehot_y = torch.zeros((z.shape[0], self.n_classes), requires_grad=False, device=self.opt.device)
                onehot_y[:, label] = 1  # Set the specified label to 1
            else:
                # No label (all zeros)
                onehot_y = torch.zeros((z.shape[0], self.n_classes),requires_grad=False, device=self.opt.device)

            # Reshape one-hot encoding for concatenation
            onehot_conv_y = onehot_y.view(z.shape[0], 1, 1, self.n_classes)
            onehot_conv_y = onehot_conv_y * torch.ones((z.shape[0], z.shape[1], z.shape[2], self.n_classes), device=self.opt.device)

            # Concatenate one-hot encoding with latent representation
            h_pos = torch.cat((z, onehot_conv_y), dim=3)

            for i, convlayer in enumerate(self.enc_model):
                h_pos = convlayer.forward(h_pos)
                activations.append(h_pos.view(h_pos.size(0), -1) )
                # print("Shape after layer",i, "encoder", h_pos.shape)

            return h_pos, onehot_y, activations, scalar_outputs
        

        

    def _latent_CVAE_mirror(self, h_pos, onehot_y, scalar_outputs, y= None, y_n= None, onehot_y_neg= None, activations = None, label = None, mode = "train"):
        
        if mode == "train":
            h_pos = utils.overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
            h_neg = utils.overlay_y_on_x3d(h_pos, y_n)

            # Flatten
            # print("Shape before latent space", h_pos.shape)
            h_pos, h_neg, nn_loss = self.fc.forward_forward(h_pos, h_neg, y)
            scalar_outputs["Loss"] += nn_loss
            h_pos.detach()
            h_neg.detach()

            mu_var = self.fc_mu_var.forward(h_pos)
            # split x in half
            mu = mu_var[:, :self.latent_dim]
            # sigma shouldn't be negative
            log_var = mu_var[:, self.latent_dim:]

            h_pos = self.reparameterize(mu, log_var)
            h_neg = torch.cat((h_pos, onehot_y_neg), dim=1)
            h_pos = torch.cat((h_pos, onehot_y), dim=1)
            # print("Shape after latent space", h_pos.shape)



            return h_pos, h_neg, mu, log_var, scalar_outputs

        elif mode == "test":

            if label is not None:
                y = torch.full((h_pos.size(0),), label, dtype=torch.long, device=h_pos.device)
                h_pos = utils.overlay_y_on_x3d(h_pos, y)
                h_pos = self.fc.forward(h_pos)
                activations.append(h_pos)
            
            else:
                h_pos = self.fc.forward(h_pos)
                
            mu_var = self.fc_mu_var.forward(h_pos)
            # split x in half
            mu = mu_var[:, :self.latent_dim]
            # sigma shouldn't be negative
            log_var = mu_var[:, self.latent_dim:]
            h_pos = self.reparameterize(mu, log_var)
            h_pos = torch.cat((h_pos, onehot_y), dim=1)
            # print("Shape after latent space", h_pos.shape)

            return h_pos, mu, log_var, scalar_outputs

        
    def _decoder_CVAE_mirror(self, h_pos, y, y_n, mu, log_var, enc_list, h_neg = None, scalar_outputs=None, mode = "train"):
        
        j=0
        if mode == "train":
            utils.overlay_y_on_x(h_pos, y)  
            utils.overlay_y_on_x(h_neg, y_n)
            h_pos, h_neg, nn_loss= self.decoder_input_0.forward_forward(h_pos, h_neg, y)
            scalar_outputs["Loss"] += nn_loss
            h_pos.detach()
            h_neg.detach()

            utils.overlay_y_on_x(h_pos, y)
            utils.overlay_y_on_x(h_neg, y_n)

            h_pos, h_neg, nn_loss = self.decoder_input_1.forward_forward(h_pos, h_neg, y)
            scalar_outputs["Loss"] += nn_loss
            h_pos.detach()
            h_neg.detach()

            h_pos = h_pos.view(-1, self.opt.VAE.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
            # Loss over the decoder fcs\
            # loss = self.elbo_loss(h_pos, enc_list[i], mu, log_var)
            # scalar_outputs["Loss"] += loss
            # h_pos.detach()
            j+=1
            for i, convlayer in enumerate(self.dec_model):
                h_pos = convlayer.forward(h_pos)
                rec_loss, kld_loss = self.elbo_loss(h_pos, enc_list[j], mu, log_var)
                scalar_outputs["Loss"] += rec_loss + kld_loss* self.beta
                h_pos.detach()
                j+=1
            scalar_outputs["MSE_loss"] = rec_loss
            scalar_outputs["KLD_loss"] = kld_loss
        elif mode == "test":
            h_pos = self.decoder_input_0.forward(h_pos)
            h_pos = self.decoder_input_1.forward(h_pos)
            h_pos = h_pos.view(-1, self.opt.VAE.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
            for i, convlayer in enumerate(self.dec_model):
                h_pos = convlayer.forward(h_pos)
                # print("Shape after layer",i, "decoder", h_pos.shape)
            
        
        
        
        if scalar_outputs is None:
            return h_pos
        else:
            return h_pos, scalar_outputs
        
    

    def _forward_FFCVAE(self, inputs, labels, scalar_outputs):
        # print(f"inputs:{inputs} \n labels:{labels}")
        torch.autograd.set_detect_anomaly(True)
        
        z = inputs

        h_pos, y, y_n, onehot_y, onehot_y_neg, enc_out, scalar_outputs = self._encoder_CVAE_mirror(z, labels, scalar_outputs)

        h_pos, h_neg, mu, log_var, scalar_outputs = self._latent_CVAE_mirror(h_pos,onehot_y, scalar_outputs, y, y_n, onehot_y_neg)

        h_pos, scalar_outputs = self._decoder_CVAE_mirror(h_pos, y, y_n,  mu, log_var, enc_out, h_neg, scalar_outputs)

        with torch.no_grad():
            scalar_outputs_test = {"Loss": torch.zeros(1, device=self.opt.device)}
            h_pos, y, y_n, onehot_y, onehot_y_neg, enc_out, scalar_outputs_test = self._encoder_CVAE_mirror(z, labels, scalar_outputs_test)
        
            h_pos, h_neg, mu, log_var, scalar_outputs_test = self._latent_CVAE_mirror(h_pos,onehot_y, scalar_outputs, y, y_n, onehot_y_neg)

        h_pos.detach()
        h_pos = self.linear_classifier(h_pos)
        loss = self.classification_loss(h_pos, labels)
        scalar_outputs["Loss"] += loss
        h_pos.detach()


        return scalar_outputs
    

    ### VFFAE ###
    def _forward_VFFAE(self, inputs, labels, scalar_outputs):

        z = inputs
        enc_out = [z]
        
        # Iterate through encoder layers
        for layer in self.enc_model:
            z, loss = layer.forward_forward(z)
            # enc_out.append(z.clone())
            scalar_outputs["Loss"] += loss
            z = z.detach()
            enc_out.append(z)
            
            

        # Reverse the encoder outputs for decoder usage
        enc_out = list(reversed(enc_out[:-1]))

        # Iterate through decoder layers
        for idx, layer in enumerate(self.dec_model):
            z, loss = layer.forward_forward(z, enc_out[idx])
            scalar_outputs["Loss"] += loss
            z = z.detach()

        if self.opt.VFFAE.classifier:
            with torch.no_grad():
                z = inputs
                for layer in self.enc_model:
                    z, _ = layer.forward_forward(z)
            z = self.linear_classifier(z)

            loss = self.classification_loss(z, labels)
            accuracy = utils.get_accuracy(self.opt, z, labels)
            scalar_outputs["Accuracy"]  = accuracy
            scalar_outputs["Loss"] += loss
        

        return scalar_outputs


    def _forward_CwC(self, inputs, labels, scalar_outputs, epoch):
        # print(f"inputs:{inputs} \n labels:{labels}")
        z = inputs
        y = labels.long()
        y_n = y[torch.randperm(z.size(0))]

        start_end = self.start_end

        # By setting show to True only every 800 iterations, it ensures that these potentially time-consuming actions 
        # don’t run in every iteration, improving efficiency.
        if self.iter % self.show_iters == 0:
            show = True
        else:
            show = False

        h_pos = z

        for i, convlayer in enumerate(self.model):
            s, e = start_end[i]
            if epoch in list(range(s, e)):
                h_pos, ff_loss = convlayer.forward_forward(h_pos, y)
                scalar_outputs["Loss"] += ff_loss
                h_pos.detach()
                if show:
                    print('Training Conv Layer: ', i, '... on epoch:', epoch)
                b1_out = h_pos.clone()
            # do not train the main model, only the classifier
            else:
                h_pos = convlayer.forward(h_pos)
        
          # Block 1 out

        ### NN LAYERS ###
        # 
        # h_pos = utils.overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
        # h_neg = utils.overlay_y_on_x3d(h_pos, y_n)

        # for i, nnlayer in enumerate(self.nn_layers):

        #     s, e = start_end[i + len(self.model) + len(self.convb2_layers)]
        #     if epoch in list(range(s, e)):
        #         h_pos, h_neg = nnlayer.forward_forward(h_pos, h_neg, y.to(self.opt.device), show)
        #         h_pos = utils.overlay_y_on_x(h_pos, y)

        #         if show:
        #             print('Training NN Layer', i, '...')
        #     else:
        #         h_pos = nnlayer(h_pos)
        #         h_neg = nnlayer(h_neg)

        #     rnd = random.randint(0, 1)
        #     if rnd == 1:
        #         h_neg = utils.overlay_y_on_x(h_neg, y_n)
        #     else:
        #         h_neg = utils.overlay_y_on_x(h_neg, y)

        if epoch >= start_end[-1][0]:
            scalar_outputs = self.predict_CwC(inputs, labels, scalar_outputs)
        self.iter += 1



        return scalar_outputs
        
    
    def forward_downstream_multi_pass(
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        # z_all = inputs["all_sample"] # bs, num_classes, C, H, W
        # z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1) # bs, num_classes, C*H*W

        # z_all = self.layer_norm(z_all)
        # input_classification_model = []

        # with torch.no_grad():
        #     for idx, layer in enumerate(self.model):
        #         z_all = layer(z_all)
        #         z_all = self.act_fn.apply(z_all)
        #         z_unnorm = z_all.clone()
        #         z_all = self.layer_norm(z_all)

        #         if idx >= 1:
        #             # print(z.shape)
        #             input_classification_model.append(z_unnorm)

        # input_classification_model = torch.concat(input_classification_model, dim=-1) # bs x 6000 # concat all activations from all layers
        # ssq_all = torch.sum(input_classification_model ** 2, dim=-1)



        z_all = inputs["all_sample"] # bs, num_classes, C, H, W
        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1) # bs, num_classes, C*H*W
        ssq_all = []
        for class_num in range(z_all.shape[1]):
            z = z_all[:, class_num, :] # bs, C*H*W
            z = self.layer_norm(z)
            input_classification_model = []

            # 784, 2000, 2000, 2000

            with torch.no_grad():
                for idx, layer in enumerate(self.model):
                    z = layer(z)
                    z = self.act_fn(z)
                    z_unnorm = z.clone()
                    z = self.layer_norm(z)

                    if idx >= 1:
                        # print(z.shape)
                        input_classification_model.append(z_unnorm)

            input_classification_model = torch.concat(input_classification_model, dim=-1) # bs x 6000 # concat all activations from all layers
            ssq = torch.sum(input_classification_model ** 2, dim=-1) # bs # sum of squares of each activation
            ssq_all.append(ssq)
        ssq_all = torch.stack(ssq_all, dim=-1) # bs x num_classes # sum of squares of each activation for each class
        
        classification_accuracy = utils.get_accuracy(
            self.opt, ssq_all.data, labels["class_labels"]
        )

        scalar_outputs["multi_pass_classification_accuracy"] = classification_accuracy
        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self.layer_norm(z) 

        input_classification_model = []

        # 784, 2000, 2000, 2000

        
        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn(z)
                z = self.layer_norm(z)

                if idx >= 1:
                    # print(z.shape)
                    input_classification_model.append(z)


        input_classification_model = torch.concat(input_classification_model, dim=-1) # concat all activations from all layers
       
        # print(input_classification_model.shape)
        # exit()

        # [0.5, 1, 1.5, ....]
        # max = 3
        # [-2.5, -2, -1.5, .. 0, ..]
        if self.opt.model.structure == "BP" or self.opt.model.structure == "BP/FF":
            output_pass = self.linear_classifier_pass(z)
            output_multi_pass = self.linear_classifier_multi_pass(input_classification_model) # bs x 10 , 
        else:
            output_pass = self.linear_classifier_pass(z.detach())
            output_multi_pass = self.linear_classifier_multi_pass(input_classification_model.detach()) # bs x 10 
            

        # z = z - torch.max(z, dim=-1, keepdim=True)[0] # not entirely clear why each entry in output is made 0 or -ve
        # output = F.softmax(z, dim=1)
        # print("Shapes: " + str(output.shape) +  str(labels["class_labels"].shape))
        # print("Values: " + str(output) +  str(labels["class_labels"]))


        # print("Output_pass:", output_pass)
        # print("Predicted classes_pass:", torch.argmax(output_pass, dim=1))
        # print("Output_pass_softmax:", F.softmax(output_pass, dim=1))
        # print("Predicted classes:", torch.argmax(F.softmax(output_pass, dim=1), dim=1))
        classification_loss_pass = self.classification_loss(output_pass, labels["class_labels"].to(self.opt.device))
        classification_accuracy_pass = utils.get_accuracy(
            self.opt, output_pass, labels["class_labels"].to(self.opt.device)
        )


        classification_loss_multi_pass = self.classification_loss(output_multi_pass, labels["class_labels"].to(self.opt.device))
        classification_accuracy_multi_pass = utils.get_accuracy(
            self.opt, output_multi_pass, labels["class_labels"].to(self.opt.device)
        )


        scalar_outputs["Loss"] += classification_loss_multi_pass
        scalar_outputs["Loss"] += classification_loss_pass
        scalar_outputs["classification_loss_multi"] = classification_loss_multi_pass
        scalar_outputs["classification_accuracy_multi"] = classification_accuracy_multi_pass
        scalar_outputs["classification_loss"] = classification_loss_pass
        scalar_outputs["classification_accuracy"] = classification_accuracy_pass
        return scalar_outputs

    def predict_CwC(self, inputs, labels, scalar_outputs = None):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }
        
        z = inputs

        h_pos = z
        with torch.no_grad():

            for i, convlayer in enumerate(self.model):
                h_pos = convlayer.forward(h_pos)
                
            ### NN LAYERS ###
            # b1_out = h_pos.clone()  # Block 1 out
            
            # h_pos = utils.overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
            # h_neg = utils.overlay_y_on_x3d(h_pos, y_n)

            # for i, nnlayer in enumerate(self.nn_layers):

            #     s, e = start_end[i + len(self.model) + len(self.convb2_layers)]
            #     if epoch in list(range(s, e)):
            #         h_pos, h_neg = nnlayer.forward_forward(h_pos, h_neg, y.to(self.opt.device), show)
            #         h_pos = utils.overlay_y_on_x(h_pos, y)

            #         if show:
            #             print('Training NN Layer', i, '...')
            #     else:
            #         h_pos = nnlayer(h_pos)
            #         h_neg = nnlayer(h_neg)

            #     rnd = random.randint(0, 1)
            #     if rnd == 1:
            #         h_neg = utils.overlay_y_on_x(h_neg, y_n)
            #     else:
            #         h_neg = utils.overlay_y_on_x(h_neg, y)
            
            
            # Step 1: Reshape tensor to [B, C, NChannels/C, H, W]
            if self.ClassGroups:
                h_reshaped = h_pos.view(h_pos.shape[0], self.n_classes[-1], self.final_channels // self.n_classes[-1], h_pos.shape[2], h_pos.shape[3])
            else:
                h_reshaped = h_pos.view(h_pos.shape[0], self.n_classes, self.final_channels // self.n_classes, h_pos.shape[2], h_pos.shape[3])

            # Step 2: Compute mean squared value for each subset
            mean_squared_values = (h_reshaped ** 2).mean(dim=[2, 3, 4])

            # print("Shapes: " + str(predicted_classes.shape) +  str(labels["class_labels"].shape))
            # print("Datatypes: " + str(predicted_classes.dtype) +  str(labels["class_labels"].dtype))
            # print("Predicted classes: " + str(predicted_classes) +  str(labels["class_labels"]))

            classification_accuracy_avg = utils.get_accuracy(self.opt, mean_squared_values.data, labels)
            classification_loss_avg = 1 - classification_accuracy_avg
            
            #classification_loss_avg = 1.0 - sf.eq(y3_p.cuda()).float().mean().item()

            scalar_outputs["classification_loss_avg"] = classification_loss_avg
            scalar_outputs["classification_accuracy_avg"] = classification_accuracy_avg

        
        if self.opt.CwC.sf_pred:
            if len(h_pos.shape) > 2:
                h_pos = h_pos.reshape(h_pos.size(0), -1)
                
            # normalize magnitude activations of previous layer to forward
            # only orientation of activity vector norm 2 sum of squared activations
            # x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
            output = self.linear_classifier(h_pos.detach())
            classification_loss_sf = self.classification_loss(output, labels)
            classification_accuracy_sf = utils.get_accuracy(self.opt, output.data, labels)

            scalar_outputs["Loss"] += classification_loss_sf #Non ha senso secondo me, se non si aumenta classification accuracy va a 0.5 sempre
            scalar_outputs["classification_loss_sf"] = classification_loss_sf
            scalar_outputs["classification_accuracy_sf"] = classification_accuracy_sf
        
        return scalar_outputs     

    def predict_AE(self, inputs, labels, visualize =False, scalar_outputs = None):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }
        
        z = inputs

        h_pos = z
        with torch.no_grad():
            for i, convlayer in enumerate(self.enc_model):
                h_pos = convlayer.forward(h_pos)
                
            for i, convlayer in enumerate(self.dec_model):
                h_pos = convlayer.forward(h_pos)

            loss = F.mse_loss(h_pos.detach(), inputs)
        
        scalar_outputs["Loss"] += loss

        if visualize:
            self.n_images = 5
            self.latent_size = self.opt.AE.channel_list[-1]
            
            print("Visualizing Autoencoder Results")
            utils.visualize_autoencoder_results(inputs, h_pos, num_images=self.n_images)

            print("Visualizing Autoencoder Generated Images")
            
            # Sample random latent vectors from a normal distribution
            latent_vectors = torch.randn(self.n_images, self.latent_size, 1, 1, device=self.opt.device)

            # Pass the latent vectors through the decoder
            h_pos = latent_vectors
            for i, convlayer in enumerate(self.dec_model):
                h_pos = convlayer.forward(h_pos)  # Assuming each decoder layer has a custom forward method `ff_infer`

            # h_pos now contains the generated images

            # Denormalize the images for visualization
            MEAN = torch.tensor([0.5], dtype=torch.float32, device=self.opt.device)
            STD = torch.tensor([0.5], dtype=torch.float32, device=self.opt.device)
            denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
            generated_images = torch.stack([denormalization(img) for img in h_pos])

            # Visualize the generated images
            utils.visualize_autoencoder_results(generated_images, num_images=self.n_images)
            
        return scalar_outputs

    
    def predict_VAE(self, inputs, labels, visualize=False, scalar_outputs=None, conditioned_generation=True, num_generate=25, grid_size=0.05):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }
        with torch.no_grad():
            z = inputs

            h_pos, y, y_n, onehot_y, onehot_y_neg, scalar_outputs = self._encoder_CVAE(z, labels, scalar_outputs)
        
            if self.latent_FF:
                h_pos, h_neg, mu, log_var, scalar_outputs = self._latent_CVAE(h_pos, y, y_n, onehot_y, onehot_y_neg, scalar_outputs)
            else:
                
                h_pos, mu, log_var, scalar_outputs = self._latent_CVAE(h_pos, y, y_n, onehot_y, onehot_y_neg, scalar_outputs)
                h_neg = None

            h_pos, scalar_outputs = self._decoder_CVAE(h_pos, y, y_n, h_neg, scalar_outputs)
    
            # Calculate losses
            rec_loss, kld_loss = self.elbo_loss(h_pos, inputs, mu, log_var)
            scalar_outputs["Loss"] += rec_loss + kld_loss * self.beta
            scalar_outputs["MSE_loss"] = rec_loss
            scalar_outputs["KLD_loss"] = kld_loss

        if visualize:
            self.n_images = min(inputs.shape[0], 5)

            # Visualization of Reconstruction
            print("Visualizing VAE Results")
            utils.visualize_autoencoder_results(inputs, h_pos, num_images=self.n_images)

            # Visualization of Mean-Reconstructed Images
            print("Visualizing VAE Mean-Reconstructed Images")
            h_pos = torch.bernoulli(h_pos)
            utils.display_and_save_batch(
                title="CVAE Reconstruction",
                batch=h_pos, # h_pos[:self.n_images]
                save=True,
                display=True
            )

            # Generate conditioned images for all labels
            if conditioned_generation and self.latent_dim == 2:
                utils.generate_images(
                        self._decoder_CVAE,
                        y.long(),
                        y_n.long(),
                        self.opt.device,
                        mode="random"
                    )
    
        return scalar_outputs
    

    def predict_FFCVAE(self, inputs, labels, visualize=False, scalar_outputs=None, conditioned_generation=True, num_generate=25, grid_size=0.05):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        # Classification task using neutral samples
        with torch.no_grad():
            z = inputs
            
            h_pos, onehot_y, _, scalar_outputs = self._encoder_CVAE_mirror(z, labels, scalar_outputs, mode = "test")
            
            mse_cnn = utils.get_prediction_CNN(h_pos, self.n_classes)

            h_pos, mu, log_var, scalar_outputs = self._latent_CVAE_mirror(h_pos, onehot_y, scalar_outputs, mode = "test")

            logits = self.linear_classifier(h_pos)
            
            loss = self.classification_loss(logits, labels)
            accuracy = utils.get_accuracy(self.opt, logits, labels)
            accuracy_cnn = utils.get_accuracy(self.opt, mse_cnn, labels)
            scalar_outputs["Accuracy CNN"]  = accuracy_cnn
            scalar_outputs["Accuracy_neutral"]  = accuracy
            scalar_outputs["Loss"] += loss

        # Classification task for each label
        ssq_all = []
        for i in range(self.n_classes):
            ssq = []
            with torch.no_grad():
                z = inputs
                
                h_pos, onehot_y, activations, scalar_outputs = self._encoder_CVAE_mirror(z, labels, scalar_outputs, label = i, mode = "test")
                
                h_pos, mu, log_var, scalar_outputs = self._latent_CVAE_mirror(h_pos, onehot_y, scalar_outputs, activations= activations, label = i, mode = "test")

                activations_concat = torch.concat(activations, dim=-1)
                
                ssq = torch.sum(activations_concat ** 2, dim=-1) 
                ssq_all.append(ssq)
        ssq_all = torch.stack(ssq_all, dim=-1)
        accuracy = utils.get_accuracy(self.opt, ssq_all, labels)
        # print(labels, ssq_all)
        # print(accuracy)
        scalar_outputs["Accuracy_multi_pass"]  = accuracy
        scalar_outputs["Loss"] += loss
        
        # Reconstruction task
        with torch.no_grad():
            z = inputs

            h_pos, y, y_n, onehot_y, onehot_y_neg, enc_out, scalar_outputs = self._encoder_CVAE_mirror(z, labels, scalar_outputs)
            
            h_pos, h_neg, mu, log_var, scalar_outputs = self._latent_CVAE_mirror(h_pos,onehot_y, scalar_outputs, y, y_n, onehot_y_neg)

            h_pos, scalar_outputs = self._decoder_CVAE_mirror(h_pos, y, y_n,  mu, log_var, enc_out, h_neg, scalar_outputs)

        if visualize:
            self.n_images = min(inputs.shape[0], 5)

            # Visualization of Reconstruction
            print("Visualizing VAE Results")
            utils.visualize_autoencoder_results(inputs, h_pos, num_images=self.n_images)

            # Visualization of Mean-Reconstructed Images
            print("Visualizing VAE Mean-Reconstructed Images")
            h_pos = torch.bernoulli(h_pos)
            utils.display_and_save_batch(
                title="CVAE Reconstruction",
                batch=h_pos, # h_pos[:self.n_images]
                save=True,
                display=True
            )

            # Generate conditioned images for all labels
            if conditioned_generation and self.latent_dim == 2:
                utils.generate_images(
                        self._decoder_CVAE,
                        y.long(),
                        y_n.long(),
                        self.opt.device,
                        mode="random"
                    )
    
        return scalar_outputs
    


    def predict_VFFAE(self, inputs, labels, visualize=False, scalar_outputs=None, conditioned_generation=True, num_generate=25, grid_size=0.05):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }
        with torch.no_grad():
            z = inputs
            enc_out = [z]
            
            # Iterate through encoder layers
            for layer in self.enc_model:
                z, loss = layer.forward_forward(z)
                scalar_outputs["Loss"] += loss
                z = z.detach()
                enc_out.append(z)

            # Reverse the encoder outputs for decoder usage
            enc_out = list(reversed(enc_out[:-1]))

            # Iterate through decoder layers
            for idx, layer in enumerate(self.dec_model):
                z, loss = layer.forward_forward(z, enc_out[idx])
                scalar_outputs["Loss"] += loss
                z = z.detach()

        if visualize:
            self.n_images = min(inputs.shape[0], 5)

            # Visualization of Reconstruction
            print("Visualizing VAE Results")
            output = z.view(-1, 1, 28, 28)
            mean = output.mean()
            stdv = output.std()
            output = (output * stdv) + mean
            utils.visualize_autoencoder_results(inputs, output, num_images=self.n_images)

            # # Visualization of Mean-Reconstructed Images
            # print("Visualizing VAE Mean-Reconstructed Images")
            # h_pos = torch.bernoulli(h_pos)
            # utils.display_and_save_batch(
            #     title="CVAE Reconstruction",
            #     batch=h_pos, # h_pos[:self.n_images]
            #     save=True,
            #     display=True
            # )

            # # Generate conditioned images for all labels
            # if conditioned_generation and self.latent_dim == 2:
            #     utils.generate_images(
            #             self._decoder_CVAE,
            #             y.long(),
            #             y_n.long(),
            #             self.opt.device,
            #             mode="random"
            #         )
    
        return scalar_outputs




# unclear as to why normal relu doesn't work
class ReLU_full_grad_ag(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
    
# Define the wrapper module for ReLU_full_grad
class ReLUFullGrad(nn.Module):
    def forward(self, input):
        # Apply the custom ReLU activation function
        return ReLU_full_grad_ag.apply(input)


class LayerNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, z):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + self.eps)


class MaxSubtractLayer(nn.Module):
    def __init__(self):
        super(MaxSubtractLayer, self).__init__()
    
    def forward(self, z):
        max_vals = torch.max(z, dim=-1, keepdim=True)[0]
        return z - max_vals
