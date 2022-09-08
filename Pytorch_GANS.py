import validation
import Short_Term_Stocks
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = Short_Term_Stocks.data_generator('Data/^IXIC.csv')


normal_y = y_train[y_train[:, 0] == 0]
normal_x = x_train[y_train[:, 0] == 0]

abnormal_x = x_train[y_train[:, 0] == 1]
abnormal_y = y_train[y_train[:, 0] == 1]

# Flatten data
normal_x = normal_x.reshape(normal_x.shape[0], -1)
normal_y = normal_y.reshape(normal_y.shape[0], -1)

abnormal_x = abnormal_x.reshape(abnormal_x.shape[0], -1)
abnormal_y = abnormal_y.reshape(abnormal_y.shape[0], -1)

torch.autograd.set_detect_anomaly(True)

normal_x_test = x_test[y_test[:, 0] == 0]
normal_y_test = y_test[y_test[:, 0] == 0]

normal_x_test = normal_x_test.reshape(normal_x_test.shape[0], -1)
normal_y_test = normal_y_test.reshape(normal_y_test.shape[0], -1)


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the Generator and Discriminator
z_dm = 64
input_s = 60 * 5
gen = Generator(input_s, z_dm).to(device)
disk = Discriminator(input_s).to(device)

criterion = nn.BCELoss()
lr = 1e-4
optimizer_disc = optim.Adam(params=disk.parameters(), lr=lr)
optimizer_gen = optim.Adam(params=gen.parameters(), lr=lr)
batch_size = 512
epochs = 50

gen_losses = []
disc_losses = []
5
for epoch in range(epochs):
    for i in range(0, len(normal_x), batch_size):
        # Initialize the x_batch and y_batch
        x_copy = normal_x[i:i + batch_size]
        y_copy = normal_y[i:i + batch_size]
        x_copy = torch.from_numpy(x_copy).float().to(device)
        y_copy = torch.from_numpy(y_copy).float().to(device)

        x_batch = normal_x[i: i + batch_size]
        y_batch = normal_y[i: i + batch_size]
        x_batch = torch.from_numpy(x_batch).float()
        y_batch = torch.from_numpy(y_batch).float()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        x_gen = gen(x_batch)
        fake = disk(x_gen)
        z = gen.encoder(x_batch)
        z_dash = gen.encoder(gen(x_batch))

        # Train the Discriminator
        # Max log(D(real)) + log(1-D(G(z)))
        loss_disc = (criterion(disk(x_batch), torch.ones_like(disk(x_batch))) +
                     criterion(fake, torch.zeros_like(fake))) / 2
        disk.zero_grad()
        loss_disc.backward(retain_graph=True)
        optimizer_disc.step()

        # Train the Generator
        # Max log(D(G(z)))
        fake = disk(x_gen)
        x_gen_copy = x_gen.cpu().detach().numpy()
        mae = nn.L1Loss()
        loss_gen = 30 * mae(x_batch, x_copy) + criterion(disk(x_gen), torch.ones_like(disk(x_gen))) + 30 * mae(z, z_dash)
        gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        print('Epoch: {}/{}'.format(epoch, epochs),
              'Iteration: {}/{}'.format(i, len(normal_x)),
              'Discriminator Loss: {:.4f}'.format(loss_disc.item()),
              'Generator Loss: {:.4f}'.format(loss_gen.item()))

        gen_losses.append(loss_gen.item())
        disc_losses.append(loss_disc.item())

# Plot the training losses
plt.figure(figsize=(10, 5))
plt.title("Training Losses on normal data")
plt.plot(gen_losses, label="Generator")
plt.plot(disc_losses, label="Discriminator")
plt.legend()
plt.show()


gen_test_losses = []


for i in range(0, len(abnormal_x)):
    x_test_gen = gen(torch.tensor(abnormal_x[i:i+1], requires_grad=False).float().to(device))
    abs_loss_gen = torch.abs(x_test_gen - torch.tensor(abnormal_x[i:i+1], requires_grad=False).float().to(device)).mean().item()

    gen_test_losses.append((criterion(disk(x_test_gen), torch.ones_like(disk(x_test_gen))) + abs_loss_gen).item())


gen_normal_test_losses = []


for i in range(0, len(normal_x_test)):
    x_normal_test_gen = gen(torch.tensor(normal_x_test[i:i+1], requires_grad=False).float().to(device))
    abs_normal_loss_gen = torch.abs(x_normal_test_gen - torch.tensor(normal_x_test[i:i+1], requires_grad=False).float().to(device)).mean().item()

    gen_normal_test_losses.append((criterion(disk(x_normal_test_gen), torch.ones_like(disk(x_normal_test_gen))) + abs_normal_loss_gen).item())


# Validation on Second Dataset
x_val1, y_val1 = validation.data_generator('Data/AAL.csv')
x_val1_norm = x_val1[y_val1[:, 0] == 0]
x_val1_ab = x_val1[y_val1[:, 0] == 1]
x_val1_norm = x_val1_norm.reshape(x_val1_norm.shape[0], -1)
x_val1_ab = x_val1_ab.reshape(x_val1_ab.shape[0], -1)


val1_gen_loss_ab = []

for i in range(0, len(x_val1_ab)):
    x_test_gen = gen(torch.tensor(x_val1_ab[i:i+1], requires_grad=False).float().to(device))
    abs_loss_gen = torch.abs(x_test_gen - torch.tensor(x_val1_ab[i:i+1], requires_grad=False).float().to(device)).mean().item()

    val1_gen_loss_ab.append((criterion(disk(x_test_gen), torch.ones_like(disk(x_test_gen))) + abs_loss_gen).item())


val1_gen_loss_norm = []


for i in range(0, len(x_val1_norm)):
    x_normal_test_gen = gen(torch.tensor(x_val1_norm[i:i+1], requires_grad=False).float().to(device))
    abs_normal_loss_gen = torch.abs(x_normal_test_gen - torch.tensor(x_val1_norm[i:i+1], requires_grad=False).float().to(device)).mean().item()

    val1_gen_loss_norm.append((criterion(disk(x_normal_test_gen), torch.ones_like(disk(x_normal_test_gen))) + abs_normal_loss_gen).item())

plt.figure(figsize=(10, 5))
plt.title("Validation Losses")
plt.plot(gen_test_losses, label="Generator_abnormal")
plt.plot(gen_normal_test_losses, label="Generator")
plt.plot(val1_gen_loss_ab, label="Val1_abnormal")
plt.plot(val1_gen_loss_norm, label="val1_normal")
plt.legend()
plt.show()