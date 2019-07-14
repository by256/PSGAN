import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from data_loaders import get_single_image_patches, load_dtd
from noise_samplers import sample_Z_l, sample_z_g
from models import SingleTextureGenerator, MultiTextureGenerator, Discriminator



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# single_image = 'honeycombed_0003.jpg'
# train = get_single_image_patches(single_image, n=50, patch_size=160, resize=None)
train = load_dtd('scaly/', n=20, patch_size=290, resize=480)

epochs = 5
batch_size = 5
n_samples = len(train)
lr = 0.0002
beta_1 = 0.5

L = 15
M = 15
d_l = 20
d_g = 40
d_p = 4
multi_texture = True

# G = SingleTextureGenerator(L, M, d_l, d_g, d_p, batch_size, device).to(device)
G = MultiTextureGenerator(L, M, d_l, d_g, d_p, batch_size, device).to(device)
G.apply(weights_init)
D = Discriminator().to(device)
D.apply(weights_init)

criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta_1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta_1, 0.999))

losses = {'d': [], 'g': [], 'D(x)': [], 'D_G(z)_d': [], 'D_G(z)_g': []}

plt.ion()
fig, axes = plt.subplots(4, 4)
axes = axes.ravel()

for epoch in range(epochs):
    for batch in range(0, n_samples, batch_size):

        # update D

        # real
        D.zero_grad()
        real_batch = torch.Tensor(train[batch:batch + batch_size]).to(device)
        real_label = torch.full((batch_size, L, M), 1, device=device)
        output = D(real_batch)
        d_real_loss = criterion(output, real_label)
        d_real_loss.backward()
        D_x = output.mean().item()

        # fake
        Z_l = torch.Tensor(sample_Z_l(batch_size, d_l, L, M)).to(device)
        z_g = torch.Tensor(sample_z_g(batch_size, d_g)).to(device)
        phi = torch.FloatTensor(batch_size, d_p, 1, 1, device=device).uniform_(0, 2*np.pi)

        if multi_texture:
            fake_samples = G(Z_l, z_g, phi)
        else:
            fake_samples = G(Z_l, phi)
        fake_label = torch.full((batch_size, L, M), 0, device=device)

        output = D(fake_samples)
        d_fake_loss = criterion(output, fake_label)
        d_fake_loss.backward(retain_graph=True)
        D_G_zd = output.mean().item()

        d_loss = d_real_loss + d_fake_loss
        
        optimizerD.step()

        # update G
        G.zero_grad()
        real_label = torch.full((batch_size, L, M), 1).to(device)
        output = D(fake_samples)

        g_loss = criterion(output, real_label)
        g_loss.backward()
        D_G_zg = output.mean().item()
        
        optimizerG.step()

    outputs = [epoch+1, epochs, d_loss.item(), g_loss.item(), D_x, D_G_zd, D_G_zg]
    outputs = [np.round(x, 3) if isinstance(x, float) else x for x in outputs]
    print('Epoch: {}/{}   D_loss: {}   G_loss: {}   D(x): {}   D(G(z)): {} - {}'.format(*outputs))

    # generate images to display
    n = 16
    root_n = int(np.sqrt(n))

    Z_l = torch.Tensor(sample_Z_l(n, d_l, L, M)).to(device)
    z_g = torch.Tensor(sample_z_g(n, d_g)).to(device)
    phi = torch.FloatTensor(n, d_p, 1, 1, device=device).uniform_(0, 2*np.pi)
    if multi_texture:
        fake_samples = G(Z_l, z_g, phi)
    else:
        fake_samples = G(Z_l, phi)
    
    if str(device) == 'cpu':
        images = fake_samples.cpu().detach().numpy()[:n, :, :, :]
    else:
        images = fake_samples.cuda().detach().numpy()[:n, :, :, :]
    images = np.transpose(images, (0, 2, 3, 1))
    images = (images*255).astype(np.uint8)
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i, :, :, ::-1])
        ax.axis('off')
    plt.show()
    plt.pause(0.00001)

    


