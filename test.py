import torch
import numpy as np
from qGAN import PatchQuantumGenerator

# Parametri modello
n_qubits = 5
n_a_qubits = 1
q_depth = 6
n_generators = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Inizializza e carica il modello
generator = PatchQuantumGenerator(
    generators=n_generators,
    qubits=n_qubits,
    ancillae=n_a_qubits,
    depth=q_depth
).to(device)

generator.load_state_dict(torch.load("generator_model.pt", map_location=device))
generator.eval()

# Generazione immagini
num_images = 10000
batch_size = 100
generated_images = []
if __name__=="__main__":
    with torch.no_grad():
        for _ in range(num_images // batch_size):
            noise = torch.rand(batch_size, n_qubits, device=device) * np.pi / 2
            fake_data = generator(noise).view(-1, 28, 28, 1)  # (B, 28, 28, 1)

            # Normalizzazione per immagine (min → 0, max → 1)
            min_vals = fake_data.amin(dim=(1, 2, 3), keepdim=True)
            max_vals = fake_data.amax(dim=(1, 2, 3), keepdim=True)
            normalized = (fake_data - min_vals) / (max_vals - min_vals + 1e-8)

            generated_images.append(normalized.cpu().numpy())

    # Concatenazione e salvataggio
    final_tensor = np.concatenate(generated_images, axis=0)  # shape: (10000, 28, 28, 1)

    np.save("generated_images.npy", final_tensor)
