import pennylane as qml
import torch
import torch.nn as nn
from typing import Any

class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_qubits:int, n_generators: int, q_delta: int =1, q_depth: int):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** (n_qubits - n_a_qubits)

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0).to(device)

        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)

        return images

class QuantumComponents:
    def __init__(self, n_qubits: int, n_a_qubits: int, q_depth: int, dev: qml.Device):
        self.n_qubits = n_qubits
        self.n_a_qubits = n_a_qubits
        self.q_depth = q_depth
        self.dev = dev

        self.qnode = qml.QNode(self.quantum_circuit, dev, diff_method="parameter-shift")

    def quantum_circuit(self, noise, weights):
        weights = weights.reshape(self.q_depth, self.n_qubits)

        for i in range(self.n_qubits):
            qml.RY(noise[i], wires=i)

        for i in range(self.q_depth):
            for y in range(self.n_qubits):
                qml.RY(weights[i][y], wires=y)
            for y in range(self.n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.probs(wires=list(range(self.n_qubits)))

    def partial_measure(self, noise, weights):
        probs = self.qnode(noise, weights)
        probsgiven0 = probs[: 2 ** (self.n_qubits - self.n_a_qubits)]
        probsgiven0 /= torch.sum(probs)
        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, q_components: QuantumComponents, q_delta=1, torch_device="cpu"):
        super().__init__()

        self.q = q_components
        self.torch_device = torch_device
        self.n_generators = n_generators

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(self.q.q_depth * self.q.n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )

    def forward(self, x):
        patch_size = 2 ** (self.q.n_qubits - self.q.n_a_qubits)
        images = torch.empty(x.size(0), 0).to(self.torch_device)

        for params in self.q_params:
            patches = torch.empty(0, patch_size).to(self.torch_device)
            for elem in x:
                q_out = self.q.partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            images = torch.cat((images, patches), 1)

        return images
