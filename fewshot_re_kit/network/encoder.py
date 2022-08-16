import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230,padding_size =1,kernel_size=3):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim
        if self.hidden_size == 0:
            self.conv = nn.Conv1d(self.embedding_dim, 1, kernel_size, padding = padding_size)
        else:
            self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, kernel_size, padding=padding_size)
        self.pool = nn.MaxPool1d(max_length+2*padding_size-kernel_size+1)

    def forward(self, inputs):
        return self.cnn(inputs)

    def cnn(self, inputs):
        x = self.conv(inputs.transpose(1, 2))#(B,hidden,length)
        x = self.pool(x)#(B,hidden,1)
        x = F.tanh(x)
        output = x.squeeze(2)
        return output

