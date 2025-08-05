import numpy as np

real = np.load("mnist_digit_1_6000.npy")
fake = np.load("generated_images.npy")
if __name__ == "__main__":
    print("Real shape:", real.shape, "Min/Max:", real.min(), real.max())
    print("Fake shape:", fake.shape, "Min/Max:", fake.min(), fake.max())