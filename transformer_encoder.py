import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, lr, input_dims, encoding_dims):
        super(PositionalEncoding, self).__init__()
        self.n_inputs = input_dims[0] * input_dims[1]

        self.encodings = nn.Linear(self.n_inputs, (self.n_inputs, encoding_dims))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs):
        encodings = self.encodings(inputs)
        encoded_vectors = T.matmul(inputs, encodings)
        return F.relu(encoded_vectors)

class Attention(nn.Module):
    def __init__(self, lr, input_dims, encoding_dims, qkv_dims):
        super(Attention, self).__init__()
        self.norm_factor = np.sqrt(qkv_dims)
        self.encoder = PositionalEncoding(lr, input_dims, encoding_dims)

        self.queries = nn.Linear((self.n_inputs, encoding_dims), qkv_dims, bias=False)
        self.keys = nn.Linear((self.n_inputs, encoding_dims), qkv_dims, bias=False)
        self.values = nn.Linear((self.n_inputs, encoding_dims), qkv_dims, bias=False)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs):
        encoded_vectors = self.encoder.forward(inputs)
        
        queries = self.queries(encoded_vectors)
        keys = self.keys(encoded_vectors)
        values = self.values(encoded_vectors)

        attention_scores = F.softmax(T.matmul(queries, keys.transpose(0, 1)) / self.norm_factor)
        attention_values = T.matmul(attention_scores, values)
        return attention_values, encoded_vectors


class MultiHeadedAttention(nn.Module):
    def __init__(self, lr, input_dims, encoding_dims, qkv_dims, n_heads):
        super(MultiHeadedAttention, self).__init__()
        self.attention_heads = [Attention(lr, input_dims, \
            encoding_dims, qkv_dims) for _ in range(n_heads)]

        fc_dims = (qkv_dims[0], qkv_dims[1]*n_heads)

        self.fc = nn.Linear(fc_dims, qkv_dims, bias=False)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs):
        attention_values = []
        skip_values = []
        for head in self.attention_heads:
            value, skip_val = head.forward(inputs)
            attention_values.append(value)
            skip_values.append(skip_val)
        mha_output = T.stack(attention_values).to(self.device)
        skip_value = skip_values[0].to(self.device)
        mha_output = self.fc(mha_output)
        return mha_output, skip_value


class TransformerEncoder(nn.Module):
    def __init__(self, lr, input_dims, encoding_dims, qkv_dims, \
        n_heads, fc1_dims, fc2_dims, output_dims):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadedAttention(lr,
            input_dims,
            encoding_dims,
            qkv_dims,
            n_heads
            )

        self.norm_1 = nn.LayerNorm(qkv_dims)

        self.feed_forward = nn.Sequential(
            nn.Linear(qkv_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, output_dims)
        )

        self.downsample = nn.Linear(qkv_dims, output_dims)

        self.norm_2 = nn.LayerNorm(output_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs):
        mha_output, skip_val = self.multi_headed_attention.forward(inputs)
        add_and_norm = self.norm_1(skip_val + mha_output)
        ff_output = self.feed_forward(add_and_norm)
        add_and_norm = self.norm_2(add_and_norm + self.downsample(ff_output))
        return ff_output