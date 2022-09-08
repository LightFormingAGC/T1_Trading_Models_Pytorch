import torch
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import Short_Term_Stocks

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train, y_train, x_test, y_test = Short_Term_Stocks.data_generator('Data/^IXIC_wk.csv')
x_train = torch.tensor(x_train, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.float).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_test = torch.tensor(y_test, dtype=torch.float).to(device)



class LSTM1(nn.Module):

    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz

        # initialize weights

        # i_t
        self.X_i = torch.randn(size=(input_sz, hidden_sz), requires_grad=True, device=device)
        self.H_i = torch.randn(size=(hidden_sz, hidden_sz), requires_grad=True, device=device)
        self.bias_i = torch.randn(size=(hidden_sz,), requires_grad=True, device=device)

        # f_t
        self.X_f = torch.randn(size=(input_sz, hidden_sz), requires_grad=True, device=device)
        self.H_f = torch.randn(size=(hidden_sz, hidden_sz), requires_grad=True, device=device)
        self.bias_f = torch.randn(size=(hidden_sz,), requires_grad=True, device=device)

        # g_t
        self.X_g = torch.randn(size=(input_sz, hidden_sz), requires_grad=True, device=device)
        self.H_g = torch.randn(size=(hidden_sz, hidden_sz), requires_grad=True, device=device)
        self.bias_g = torch.randn(size=(hidden_sz,), requires_grad=True, device=device)

        # o_t
        self.X_o = torch.randn(size=(input_sz, hidden_sz), requires_grad=True, device=device)
        self.H_o = torch.randn(size=(hidden_sz, hidden_sz), requires_grad=True, device=device)
        self.bias_o = torch.randn(size=(hidden_sz,), requires_grad=True, device=device)

        # final layer
        self.W_out = torch.randn(size=(hidden_sz, 3), requires_grad=True, device=device)
        self.bias_out = torch.randn(size=(3,), requires_grad=True, device=device)

        self.params = [self.X_i, self.H_i, self.bias_i, self.X_f, self.H_f, self.bias_f, self.X_g, self.H_g,
                       self.bias_g, self.X_o, self.H_o, self.bias_o, self.W_out, self.bias_out]

    def init_hidden_states(self, batch_sz):
        h_t, c_t = torch.zeros(size=(batch_sz, self.hidden_size), device=device), torch.zeros(size=(batch_sz, self.hidden_size), device=device)
        return h_t, c_t

    def forward(self, x, initial_hidden_states):
        batch_s, seq_s, _ = x.size()
        hidden_seq = []
        if initial_hidden_states is None:
            h_t, c_t = self.init_hidden_states(batch_s)
        else:
            h_t, c_t = initial_hidden_states

        for t in range(seq_s):

            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.X_i + h_t @ self.H_i + self.bias_i).to(device)
            f_t = torch.sigmoid(x_t @ self.X_f + h_t @ self.H_f + self.bias_f).to(device)
            g_t = torch.tanh(x_t @ self.X_g + h_t @ self.H_g + self.bias_g).to(device)
            o_t = torch.sigmoid(x_t @ self.X_o + h_t @ self.H_o + self.bias_o).to(device)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            c_t.to(device)
            h_t.to(device)

            final_out = h_t @ self.W_out + self.bias_out
            final_out = torch.softmax(final_out, dim=1)
            hidden_seq.append(final_out.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0).to(device)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)


lstm = LSTM1(input_sz=x_train.size()[-1], hidden_sz=15).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm.params, lr=0.01)
num_epochs = 10


hidden_s = None
losses = []
val_losses = []
mean_batch_loss = 10
for epoch in range(num_epochs):
    batch_losses = []
    for batch in np.arange(0, x_train.size()[0], 5):
        x_batch = x_train[batch:batch + 100, :, :]
        y_batch = y_train[batch:batch + 100, :, :]
        hidden_seq, _ = lstm.forward(x_batch, hidden_s)
        loss = criterion(hidden_seq, y_batch)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(loss.item())

        # validation loss
        x_val = x_test[:, :, :]
        y_val = y_test[:, :, :]
        hidden_seq, _ = lstm.forward(x_val, hidden_s)
        val_loss = criterion(hidden_seq, y_val)
        val_losses.append(val_loss.item())

        batch_losses.append(loss.item())

        if batch % 10 == 0:
            print('Epoch: {}/{}'.format(epoch, num_epochs),
                  'Loss: {}'.format(loss.item()), 'Val Loss: {}'.format(val_loss.item()))
    temp_batch_losses = np.mean(batch_losses)
    if temp_batch_losses < mean_batch_loss:
        mean_batch_loss = temp_batch_losses
    else:
        break

test_outputs = []
for batch in np.arange(0, x_test.size()[0], 10):
    x_batch = x_test[batch:batch + 10, :, :]
    y_batch = y_test[batch:batch + 10, :, :]
    hidden_seq, _ = lstm.forward(x_batch, hidden_s)
    pred = hidden_seq[:, -1, :]
    for ele in range(pred.size()[0]):
        if pred[ele, 0] == max(pred[ele, :]):
            pred[ele, 0] = 1
        else:
            pred[ele, 0] = 0
    test_outputs.append(pred)


# plot the losses
plt.plot(losses)
plt.plot(val_losses)
plt.legend(['Training', 'Test'])
plt.show()

test_outputs = torch.cat(test_outputs, dim=0)
print(test_outputs)
print(len(test_outputs))













