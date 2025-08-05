from absl import app
from absl import flags
import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("workdir0", None, "Path to the real images (.npy file).")
flags.DEFINE_string("workdir1", None, "Path to the fake images (.npy file).")
flags.DEFINE_integer("channels", 1, "Number of image channels.")
flags.DEFINE_integer("height", 28, "Height of images.")
flags.DEFINE_integer("width", 28, "Width of images.")

# Required
flags.mark_flags_as_required(['workdir0', 'workdir1'])


def calculate_is(fake_torch):
    inception = InceptionScore()
    for i in range(0, len(fake_torch), 100):
        batch = fake_torch[i:i+100].to(torch.uint8).cpu()
        inception.update(batch)
    return inception.compute()


def calculate_kid(real_torch, fake_torch):
    kid = KernelInceptionDistance()
    for i in range(0, len(fake_torch), 100):
        fake_batch = fake_torch[i:i+100].to(torch.uint8).cpu()
        real_batch = real_torch[i:i+100].to(torch.uint8).cpu()
        kid.update(real_batch, real=True)
        kid.update(fake_batch, real=False)
    return kid.compute()


def calculate_fid(real_torch, fake_torch):
    fid = FrechetInceptionDistance(feature=2048)
    for i in range(0, len(fake_torch), 100):
        fake_batch = fake_torch[i:i+100].to(torch.uint8).cpu()
        real_batch = real_torch[i:i+100].to(torch.uint8).cpu()
        fid.update(real_batch, real=True)
        fid.update(fake_batch, real=False)
    return fid.compute()


def main(argv):
    # Load numpy arrays
    real_np = np.load(FLAGS.workdir0)
    fake_np = np.load(FLAGS.workdir1)

    # Shape parameters
    c, h, w = FLAGS.channels, FLAGS.height, FLAGS.width

    # Convert to torch tensors, rescale, reshape
    real_images = torch.from_numpy(np.clip(255 * real_np + 0.5, 0, 255)).view(-1, c, h, w)
    fake_images = torch.from_numpy(np.clip(255 * fake_np + 0.5, 0, 255)).view(-1, c, h, w)
    real_tensor = torch.tile(real_images, (3, 1, 1)) if c == 1 else real_images
    fake_tensor = torch.tile(fake_images, (3, 1, 1)) if c == 1 else fake_images
    # Compute metrics
    fid = calculate_fid(real_tensor, fake_tensor)
    isc = calculate_is(fake_tensor)
    kid = calculate_kid(real_tensor, fake_tensor)

    print(f"Metrics:\n  FID = {fid}\n  KID = {kid}\n  IS  = {isc}")


if __name__ == '__main__':
    app.run(main)

