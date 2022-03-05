import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, encoding_dims=768, first_block=False):
        super(PositionalEncoding, self).__init__()
        self.first_block = first_block

        self.encodings = nn.Conv2d(
            in_channels=1,
            out_channels=encoding_dims,
            kernel_size=1,
            stride=1
            )

    def forward(self, inputs):
        if not self.first_blockfirst_block:
            return inputs
        ## input_dims (N, in_channels, height, width)
        encodings = self.encodings(inputs)
        encoded_vectors = T.transpose(encodings.flatten(start_dim=2), 1, 2)
        ## encoded dims (N, input_dims[-2] * input_dims[-1], encoding_dims)
        return encoded_vectors

class Attention(nn.Module):
    def __init__(self, encoding_dims, first_block=False):
        super(Attention, self).__init__()
        self.norm_factor = np.sqrt(encoding_dims)
        self.encoder = PositionalEncoding(encoding_dims, first_block)

        self.queries = nn.Linear(encoding_dims, encoding_dims, bias=False)
        self.keys = nn.Linear(encoding_dims, encoding_dims, bias=False)
        self.values = nn.Linear(encoding_dims, encoding_dims, bias=False)

    def forward(self, inputs):
        encoded_vectors = self.encoder.forward(inputs)

        ## encoded dims (N, input_dims[-2] * input_dims[-1], encoding_dims)
        queries = self.queries(encoded_vectors)
        keys = self.keys(encoded_vectors)
        values = self.values(encoded_vectors)
        ## qkv dims (N, input_dims[-2] * input_dims[-1], encoding_dims)

        out = T.einsum('tuv, tvw -> tuw', queries, keys.transpose(-2, -1)) / self.norm_factor
        out = F.softmax(out, dim=-1)
        ## out dims (N, input_dims[-2] * input_dims[-1], input_dims[-2] * input_dims[-1])
        attention_values = T.einsum('tuu, tuv -> tuv', out, values)
        ## att val dims == encoded dims
        return attention_values, encoded_vectors


class MultiHeadedAttention(nn.Module):
    def __init__(self, encoding_dims, n_heads, first_block=False):
        super(MultiHeadedAttention, self).__init__()
        self.attention_heads = [Attention(encoding_dims, first_block=False) \
            for _ in range(n_heads)]

        self.fc = nn.Linear(encoding_dims, encoding_dims, bias=False)

    def forward(self, inputs):
        attention_values = []
        for head in self.attention_heads:
            value, skip_val = head.forward(inputs)
            attention_values.append(value.reshape(1, *value.shape))
        mha_output = T.stack(attention_values).mean(dim=0)
        ## mha_out dims (N, input_dims[-2] * input_dims[-1], encoding_dims)
        skip_value = skip_val
        mha_output = self.fc(mha_output)
        return mha_output, skip_value


class TransformerEncoder(nn.Module):
    def __init__(self, encoding_dims, n_heads, \
        fc1_dims, fc2_dims, output_dims, first_block=False):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadedAttention(
            encoding_dims,
            n_heads,
            first_block
            )

        self.norm_1 = nn.LayerNorm(encoding_dims)

        self.feed_forward = nn.Sequential(
            nn.Linear(encoding_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, encoding_dims)
        )

        self.norm_2 = nn.LayerNorm(output_dims)

    def forward(self, inputs):
        mha_output, skip_val = self.multi_headed_attention.forward(inputs)
        add_and_norm = self.norm_1(skip_val + mha_output)
        ff_output = self.feed_forward(add_and_norm)
        output = self.norm_2(add_and_norm + ff_output)
        ## output dims (N, input_dims[-2]*input_dims[-1], encoding_dim)
        return output

class TransformerNetwork(nn.Module):
    def __init__(self, lr, encoding_dims, n_heads, \
        fc1_dims, fc2_dims, output_dims, n_encoder_blocks):
        super(TransformerNetwork, self).__init__()
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(encoding_dims, n_heads, fc1_dims, \
                fc2_dims, output_dims, first_block=True)] + \
            [TransformerEncoder(encoding_dims, n_heads, fc1_dims, \
                fc2_dims, output_dims) for _ in range(n_encoder_blocks-1)]
        )
        self.network = nn.Sequential(
            *self.encoder_blocks,
            nn.LayerNorm(encoding_dims)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        return self.network(input)