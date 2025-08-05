import pennylane as qml
import torch
import torch.nn as nn
from typing import Any


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method."""

    def __init__(self,
                 generators: int,
                 qubits: int,
                 ancillae: int,
                 depth: int,
                 q_delta: float = 1.0):
        """
        Args:
            generators (int): Number of sub-generators to be used in the patch method.
            qubits (int): Number of qubits to be used in the patch method.
            ancillae (int): Number of ancilla qubits to be used in the patch method.
            depth (int): Number layers of the quantum circuit.
            q_delta (float): Spread of the random distribution for parameter initialization.
        """
        super().__init__()
        self.generators = generators
        self.n_qubits = qubits
        self.n_a_qubits = ancillae
        self.q_depth = depth
        self.post_upscale = nn.Linear(self.generators * (2 ** (self.n_qubits - self.n_a_qubits)), 784)
        # Quantum simulator
        dev = qml.device("lightning.qubit", wires=self.n_qubits)

        # Cuda device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize parameters for each sub-generator
        self.q_params = nn.ParameterList([
            nn.Parameter(q_delta * torch.rand(self.q_depth * self.n_qubits), requires_grad=True)
            for _ in range(generators)
        ])

        # Define the internal quantum circuit as a QNode
        self._qnode = qml.QNode(self._circuit, dev, interface="torch", diff_method="parameter-shift")

    def _circuit(self, noise: torch.Tensor, weights: torch.Tensor):
        """Quantum circuit architecture"""
        weights = weights.reshape(self.q_depth, self.n_qubits)

        # Encode latent vector
        for i in range(self.n_qubits):
            qml.RY(noise[i], wires=i)

        # Parameterised layers
        for i in range(self.q_depth):
            for y in range(self.n_qubits):
                qml.RY(weights[i][y], wires=y)
            for y in range(self.n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.probs(wires=list(range(self.n_qubits)))

    def __partial_measure(self, noise: torch.Tensor, weights: torch.Tensor):
        """Partial measurement for non-linearity and post-processing"""
        probs = self._qnode(noise, weights)
        probsgiven0 = probs[: (2 ** (self.n_qubits - self.n_a_qubits))]
        probsgiven0 /= torch.sum(probs)
        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the quantum patch generator.
        Args:
            x: Input latent vectors of shape (batch_size, n_qubits)
        Returns:
            Tensor: Generated patches/images of shape (batch_size, total_output_size)
        """
        patch_size = 2 ** (self.n_qubits - self.n_a_qubits)
        batch_size = x.size(0)
        images = torch.Tensor(batch_size, 0).to(self.device)

        for params in self.q_params:
            patches = torch.Tensor(0, patch_size).to(self.device)
            for elem in x:
                q_out = self.__partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out), dim=0)
            images = torch.cat((images, patches), dim=1)
        return self.post_upscale(images)
        #return images

import torch
import math
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import matplotlib.gridspec as gridspec
from torch import optim
from torch import nn

if __name__ == "__main__":
    # Quantum variables
    n_qubits = 5  # Total number of qubits / N
    n_a_qubits = 1  # Number of ancillary qubits / N_A
    q_depth = 6  # Depth of the parameterised quantum circuit / D
    n_generators = 4  # Number of subgenerators for the patch method / N_G

    # ======= PARAMETRI TEST =======
    batch_size = 8
    lrG = 0.01
    num_iter = 10  # simuliamo fino a 250 iterazioni come esempio
    image_size = 28  # output = 784 → reshape in (28x28)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ======= INIZIALIZZA IL MODELLO =======
    generator = PatchQuantumGenerator(
        generators=n_generators,
        qubits=n_qubits,
        ancillae=n_a_qubits,
        depth=q_depth
    ).to(device)

    optG = optim.SGD(generator.parameters(), lr=lrG)

    # ======= PREPARA NOISE FISSO PER MONITORAGGIO =======
    fixed_noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2

    # ======= RISULTATI DA PLOTTARE =======
    results = []

    # ======= LOOP DI TEST SOLO SUL GENERATORE =======
    print("Testing PatchQuantumGenerator...")

    for counter in range(1, num_iter + 1):
        generator.zero_grad()

        # Noise casuale per ogni batch
        noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
        fake_data = generator(noise)

        # Simuliamo un "falso training step" solo per test
        loss = fake_data.mean()  # finta loss, non ha effetto pratico
        loss.backward()
        optG.step()

        # Ogni 10 iterazioni mostra log
        if counter % 10 == 0:
            print(f"Iterazione: {counter}, Output shape: {fake_data.shape}")

        # Ogni 50 iterazioni salviamo output per visualizzazione
        if counter % 10 == 0:
            with torch.no_grad():
                gen_images = generator(fixed_noise).view(batch_size, 1, image_size, image_size)
                results.append(gen_images.cpu())

    # ======= VISUALIZZAZIONE =======
    print("Visualizzazione risultati...")

    fig = plt.figure(figsize=(10, 5))
    outer = gridspec.GridSpec(len(results), 1, wspace=0.1)

    for i, images in enumerate(results):
        inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0), subplot_spec=outer[i])
        images = torch.squeeze(images, dim=1)

        for j, im in enumerate(images):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(im.numpy(), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_title(f'Iteration {50 + i * 50}', loc='left')
            fig.add_subplot(ax)

    plt.show()
