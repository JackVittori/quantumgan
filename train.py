from data_utils import process_mnist
from qGAN import Discriminator, PatchQuantumGenerator
# Library imports
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from datetime import datetime

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Hyperparmeters
lrG = 0.3  # Learning rate for the generator
lrD = 0.01  # Learning rate for the discriminator
# number of epochs
num_epochs = 20
image_size = 28  # Height / width of the square images
batch_size = 1  # Batch size
n_qubits = 5  # Total number of qubits / N
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 6  # Depth of the parameterised quantum circuit / D
n_generators = 4  # Number of subgenerators for the patch method / N_G
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

discriminator = Discriminator().to(device)
generator = PatchQuantumGenerator(
        generators=n_generators,
        qubits=n_qubits,
        ancillae=n_a_qubits,
        depth=q_depth
    ).to(device)

# Binary cross entropy
criterion = nn.BCELoss()

# Optimisers
optD = optim.SGD(discriminator.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

# Fixed noise allows us to visually track the generated images throughout training
fixed_noise = torch.rand(8, n_qubits, device=device) * math.pi / 2

# Iteration counter
counter = 0
d_losses = []
g_losses = []

# Collect images for plotting later
results = []

train_dataloader, _, _ = process_mnist(batch_size=1, selected_labels=[1])

if __name__ == '__main__':
    for epoch in range(1, num_epochs + 1):
        seen_elements = 0

        for i, (data, _) in enumerate(train_dataloader):
            data = data.reshape(-1, image_size * image_size)
            real_data = data.to(device)

            noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
            fake_data = generator(noise)

            discriminator.zero_grad()
            out_real = discriminator(real_data).view(-1)
            out_fake = discriminator(fake_data.detach()).view(-1)

            errD_real = criterion(out_real, real_labels)
            errD_fake = criterion(out_fake, fake_labels)
            errD_real.backward()
            errD_fake.backward()
            errD = errD_real + errD_fake
            optD.step()

            generator.zero_grad()
            out_fake = discriminator(fake_data).view(-1)
            errG = criterion(out_fake, real_labels)
            errG.backward()
            optG.step()

            seen_elements += data.size(0)

            if seen_elements % 500 == 0:
                print(f'[Epoch {epoch}] Seen {seen_elements} elements – D Loss: {errD:.3f}, G Loss: {errG:.3f}')
                test_images = generator(fixed_noise).view(8, 1, image_size, image_size).cpu().detach()
                results.append((epoch, seen_elements, test_images))
        # === Save model ===
    torch.save(generator.state_dict(), 'generator_model.pt')
    torch.save(discriminator.state_dict(), 'discriminator_model.pt')
    print("Models saved to generator_model.pt and discriminator_model.pt")

    # === Plot and save image grid ===
    fig = plt.figure(figsize=(12, len(results) * 2))
    outer = gridspec.GridSpec(len(results), 1, wspace=0.2, hspace=0.4)

    for idx, (epoch, seen, images) in enumerate(results):
        inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0),
                                                 subplot_spec=outer[idx])
        images = torch.squeeze(images, dim=1)

        for j, im in enumerate(images):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(im.numpy(), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_title(f"Epoch {epoch} – Seen {seen} elements", loc='left', fontsize=10)
            fig.add_subplot(ax)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"generated_images_{timestamp}.png"
    fig.savefig(image_filename)
    plt.close(fig)
    print(f"Generated images saved to {image_filename}")

