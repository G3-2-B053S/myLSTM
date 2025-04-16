import torch
import torch.nn as nn
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        #self.input_gate = nn.Sequential(
            # (batch_size, hidden_size)
        #    nn.Linear(input_size + hidden_size, hidden_size),
            # (batch_size, input_size+hidden_size)
        #    nn.Hardsigmoid()
        #)

        self.forget_gate = nn.Sequential(
            # (batch_size, hidden_size)
            nn.Linear(input_size + hidden_size, hidden_size),
            # (batch_size, input_size+hidden_size)
            nn.Sigmoid()
        )

        self.output_gate = nn.Sequential(
            # (batch_size, hidden_size)
            nn.Linear(input_size + hidden_size, hidden_size),
            # (batch_size, input_size+hidden_size)
            nn.Sigmoid()
        )

        self.c_hat = nn.Sequential(
            # (batch_size, hidden_size)
            nn.Linear(input_size + hidden_size, hidden_size),
            # (batch_size, input_size+hidden_size)
            nn.Tanh()
        )

    def forward(self, x_t, hc_t_1):
        # x_t.shape = (batch_size,input_size)
        # h_t_1.shape = c_t_1.shape=(batch_size,hidden_size)
        h_t_1, c_t_1 = hc_t_1

        # xh.shape=(batch_size,input_size+hidden_size)
        xh = torch.cat((x_t, h_t_1), dim=1)
        #print(x_t.shape,h_t_1.shape)
        #print(xh.shape)
        #i_t = self.input_gate(xh)
        f_t = self.forget_gate(xh)
        i_t=1-f_t
        o_t = self.output_gate(xh)
        c_hat = self.c_hat(xh)
        #print(f_t.shape,i_t.shape)
        c_t = f_t * c_t_1 + i_t * c_hat
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class LSTM(nn.Module):
    # x (batch_size,seq_len,input_size)
    # h (batch_size,seq_len,hidden_size)
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.cell = LSTMCell(input_size, hidden_size)
        #self.cell2 = LSTMCell(input_size, hidden_size)
        #self.nextlayer = nn.Linear(hidden_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, hc_0=None):  # (h_0,c_0)
        # x.shape=(batch_size,seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        hidden_size = self.hidden_size

        cell = self.cell
        if hc_0 is not None:
            h_0, c_0 = hc_0
        else:
            h_0 = torch.zero(batch_size, hidden_size)
            c_0 = torch.zero(batch_size, hidden_size)

        h_t_1 = h_0
        c_t_1 = c_0
        #h_t_2 = h_0
        #c_t_2 = c_0
        h_list = []

        for t in range(seq_len):
            # x_t.shape = (batch_size ,1, input_size)
            x_t = x[:, t, :]
            h_t1, c_t1 = cell(x_t, (h_t_1, c_t_1))
            #print(h_t.shape,x_t.shape,x.shape)
            #x2=self.nextlayer(h_t1)
            #h_t2, c_t2 = self.cell2(x2, (h_t_2, c_t_2))
            h_list.append(h_t1)
            h_t_1, c_t_1 = h_t1, c_t1
            #h_t_2, c_t_2 = h_t2, c_t2

        # h_list = [(batch_size,hidden_size)*seq_len]
        # [(batch_size,1,hidden_size)]
        # h.shape=(batch_size,seq_len,hidden_size)

        h = torch.stack(h_list, dim=1)
        return h, h_t_1, c_t_1

class myLSTM(nn.Module):
    # x (batch_size,seq_len,input_size)
    # h (batch_size,seq_len,hidden_size)
    def __init__(self, input_size, hidden_size,num_layers):
        super(myLSTM, self).__init__()
        self.lstm = LSTM(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hc_0=None):  # (h_0,c_0)
        batch_size, _, _ = x.size()
        if hc_0 is not None:
            h_0, c_0 = hc_0
        else:
            h_0 = torch.zeros(batch_size, self.hidden_size)
            c_0 = torch.zeros(batch_size, self.hidden_size)

        h_t_1 = h_0
        c_t_1 = c_0
        h =x
        #for i in range(self.num_layers):
        h, h_t_1, c_t_1=self.lstm(h,(h_t_1,c_t_1))

        return h,(h_t_1,c_t_1)
