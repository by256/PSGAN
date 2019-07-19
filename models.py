import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from scipy import interpolate

from noise_samplers import sample_Z_l, sample_Z_g


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def weights_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(self.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(self.weight.data, 1.0, 0.02)
            nn.init.constant_(self.bias.data, 0)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.eval()
    

class SingleTextureGenerator(BaseModel):

    def __init__(self, L, M, d_l, d_g, d_p, batch_size, device):
        super(SingleTextureGenerator, self).__init__()
        # args
        self.L = L
        self.M = M
        self.d_l = d_l
        self.d_g = d_g
        self.d_p = d_p
        self.batch_size = batch_size
        self.device = device
        # periodic mlp layer ops
        self.b1 = nn.Parameter(torch.FloatTensor(self.d_p).normal_(1, 0.02), requires_grad=True)
        self.b2 = nn.Parameter(torch.FloatTensor(self.d_p).normal_(1, 0.02), requires_grad=True)
        # generator layer ops
        self.deconv1 = nn.ConvTranspose2d(self.d_l+self.d_p, 512, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 5, 2, 2)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 5, 2, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 5, 2, 2)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, 5, 2, 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # initialise weights
        self.weights_init()


    def forward(self, Z_l, phi):
        # periodic mlp
        K = torch.cat((self.b1.unsqueeze(0), self.b1.unsqueeze(0)), dim=0)
        K = K.view(-1, self.d_p, 2)
        # zeta
        lmbda = torch.arange(self.L)
        mu = torch.arange(self.M)
        xx, yy = torch.meshgrid(lmbda, mu)
        grid = torch.stack((xx, yy), dim=0).to(torch.float).to(self.device)
        x = torch.tensordot(K, grid, dims=([2], [0]))
        phi = phi.repeat(1, 1, self.L, self.M)
        Z_p = torch.sin(x + phi)

        # generator
        Z = torch.cat((Z_l, Z_p), dim=1)
        x = self.relu(self.bn1(self.deconv1(Z)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))
        return x

    def generate_samples(self, n, detach=True):
        Z_l = torch.Tensor(sample_Z_l(n, self.d_l, self.L, self.M)).to(self.device)
        phi = torch.FloatTensor(n, self.d_p, 1, 1, device=self.device).uniform_(0.0, 2*np.pi)
        samples = self(Z_l, phi)
        if detach:
            if str(self.device) == 'cpu':
                samples = samples.cpu().detach().numpy()
            else:
                samples = samples.cuda().detach().numpy()
            samples = np.transpose(samples, (0, 2, 3, 1))
            samples = (samples*255).astype(np.uint8)
        return samples


class MultiTextureGenerator(BaseModel):

    def __init__(self, L, M, d_l, d_g, d_p, batch_size, device):
        super(MultiTextureGenerator, self).__init__()
        # args
        self.L = L
        self.M = M
        self.d_l = d_l
        self.d_g = d_g
        self.d_p = d_p
        self.batch_size = batch_size
        self.device = device
        # periodic mlp layer ops
        self.linear1 = nn.Linear(self.d_g, 60)
        self.linear2 = nn.Linear(60, self.d_p)
        self.linear3 = nn.Linear(60, self.d_p)
        # generator layer ops
        self.deconv1 = nn.ConvTranspose2d(self.d_l+self.d_g+self.d_p, 512, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 5, 2, 2)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 5, 2, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 5, 2, 2)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, 5, 2, 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # initialise weights
        self.weights_init()

    
    def forward(self, Z_l, Z_g, phi):
        z_g = Z_g[:, :, 0, 0]
        # periodic mlp
        inner_K = self.relu(self.linear1(z_g))
        K1 = self.linear2(inner_K)
        K2 = self.linear3(inner_K)
        K = torch.cat((K1, K2), dim=0)
        K = K.view(-1, self.d_p, 2)
        # zeta
        lmbda = torch.arange(self.L)
        mu = torch.arange(self.M)
        xx, yy = torch.meshgrid(lmbda, mu)
        grid = torch.stack((xx, yy), dim=0).to(torch.float).to(self.device)
        x = torch.tensordot(K, grid, dims=([2], [0]))
        phi = phi.repeat(1, 1, self.L, self.M)
        Z_p = torch.sin(x + phi)

        # generator
        Z = torch.cat((Z_l, Z_g, Z_p), dim=1)
        x = self.relu(self.bn1(self.deconv1(Z)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))
        return x


    def generate_samples(self, n, detach=True):
        Z_l = torch.Tensor(sample_Z_l(n, self.d_l, self.L, self.M)).to(self.device)
        z_g = torch.Tensor(sample_z_g(n, self.d_g)).to(self.device)
        
        phi = torch.FloatTensor(n, self.d_p, 1, 1, device=self.device).uniform_(0.0, 2*np.pi)
        samples = self(Z_l, z_g, phi)
        if detach:
            if str(self.device) == 'cpu':
                samples = samples.cpu().detach().numpy()
            else:
                samples = samples.cuda().detach().numpy()
            samples = np.transpose(samples, (0, 2, 3, 1))
            samples = (samples*255).astype(np.uint8)
        return samples

    def morph_textures(self):
        Z_l = torch.Tensor(sample_Z_l(self.batch_size, self.d_l, self.L, self.M)).to(self.device)
        z_g = sample_z_g(self.batch_size, self.d_g)
        Z_g = Z_g.repeat(1, 1, self.L, self.M)

        print(Z_l.shape)
        corners = []
        for i in range(4):
            corner = np.random.uniform(size=(-1.0, 1.0, self.d_g,))
            corners.append(corner)
        
        points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid = []
        nn = 40
        for i in range(40):
            for j in range(40):
                grid += [np.array([i / (40 - 1.0), j / (40 - 1.0)])]

        morph = interpolate.griddata(points, corners, grid, method='linear')
        return morph


class Discriminator(BaseModel):

    def __init__(self, device):
        super(Discriminator, self).__init__()
        self.device = device
        # discriminator layer ops
        self.conv1 = nn.Conv2d(3, 64, 5, 2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.conv3 = nn.Conv2d(128, 256, 5, 2, 2)
        self.conv4 = nn.Conv2d(256, 512, 5, 2, 2)
        self.conv5 = nn.Conv2d(512, 1, 5, 2, 2)

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        # initialise weights
        self.weights_init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x