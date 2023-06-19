#Simple GAN using Fully Connected layers
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, output_size):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, output_size),
            nn.Tanh()  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 3e-4
noise_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50

# Initialize discriminator and generator
discriminator = Discriminator(image_dim).to(device)
generator = Generator(noise_dim, image_dim).to(device)

# Create fixed noise for visualization during training
fixed_noise = torch.randn((batch_size, noise_dim)).to(device)

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizers
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# Tensorboard writer
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.view(-1, image_dim).to(device)
        batch_size = real_images.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(noise)
        
        disc_real = discriminator(real_images).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        
        disc_fake = discriminator(fake_images).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        loss_discriminator = (loss_disc_real + loss_disc_fake) / 2
        discriminator.zero_grad()
        loss_discriminator.backward(retain_graph=True)
        optimizer_discriminator.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = discriminator(fake_images).view(-1)
        loss_generator = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        loss_generator.backward()
        optimizer_generator.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_discriminator:.4f}, Loss G: {loss_generator:.4f}"
            )

            with torch.no_grad():
                fake_images = generator(fixed_noise).reshape(-1, 1, 28, 28)
                real_images = real_images.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake_images, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_images, normalize=True)

                writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)
                step += 1

