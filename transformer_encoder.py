import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, input_dims, encoding_dims=768, first_block=False):
        super(PositionalEncoding, self).__init__()
        self.first_block = first_block
        input_dims = input_dims[-2] * input_dims[-1]
        if first_block:
            self.cls_token = nn.Parameter(T.zeros(1, 1, encoding_dims))
            self.pos_embed = nn.Parameter(T.zeros(1, 1 + input_dims, \
                encoding_dims))

        self.encodings = nn.Conv2d(
            in_channels=1,
            out_channels=encoding_dims,
            kernel_size=1,
            stride=1
            )

    def forward(self, inputs):
        if not self.first_block:
            return inputs
        ## input_dims (N, in_channels, height, width)
        encodings = self.encodings(inputs)
        encoded_vectors = T.transpose(encodings.flatten(start_dim=2), 1, 2)
        ## encoded dims (N, input_dims[-2] * input_dims[-1], encoding_dims)
        cls_token = self.cls_token.expand(inputs.shape[0], -1, -1)
        encoded_vectors = T.cat((cls_token, encoded_vectors), dim=1)
        encoded_vectors += self.pos_embed
        ## encoded dims (N, 1 + (input_dims[-2] * input_dims[-1]), encoding_dims)
        return encoded_vectors

class Attention(nn.Module):
    def __init__(self, input_dims, encoding_dims, first_block=False):
        super(Attention, self).__init__()
        self.norm_factor = np.sqrt(encoding_dims)
        self.encoder = PositionalEncoding(input_dims, encoding_dims, first_block)

        self.queries = nn.Linear(encoding_dims, encoding_dims, bias=False)
        self.keys = nn.Linear(encoding_dims, encoding_dims, bias=False)
        self.values = nn.Linear(encoding_dims, encoding_dims, bias=False)

    def forward(self, inputs):
        encoded_vectors = self.encoder.forward(inputs)

        ## encoded dims (N, 1 + (input_dims[-2] * input_dims[-1]), encoding_dims)
        queries = self.queries(encoded_vectors)
        keys = self.keys(encoded_vectors)
        values = self.values(encoded_vectors)
        ## qkv dims (N, 1 + (input_dims[-2] * input_dims[-1]), encoding_dims)

        out = T.einsum('tuv, tvw -> tuw', queries, keys.transpose(-2, -1)) / self.norm_factor
        out = F.softmax(out, dim=-1)
        ## out dims (N, 1+ (input_dims[-2] * input_dims[-1]), 1 + (input_dims[-2] * input_dims[-1]))
        attention_values = T.einsum('tuu, tuv -> tuv', out, values)
        ## att val dims == encoded dims
        return attention_values, encoded_vectors


class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dims, encoding_dims, n_heads, first_block=False):
        super(MultiHeadedAttention, self).__init__()
        self.attention_heads = [Attention(input_dims, encoding_dims, first_block=first_block) \
            for _ in range(n_heads)]

        self.fc = nn.Linear(encoding_dims, encoding_dims, bias=False)

    def forward(self, inputs):
        attention_values = []
        for head in self.attention_heads:
            value, skip_val = head.forward(inputs)
            attention_values.append(value)
        mha_output = T.stack(attention_values).mean(dim=0)
        ## mha_out dims (N, 1 + (input_dims[-2] * input_dims[-1]), encoding_dims)
        skip_value = skip_val
        mha_output = self.fc(mha_output)
        return mha_output, skip_value


class TransformerEncoder(nn.Module):
    def __init__(self, input_dims, encoding_dims, n_heads, \
        fc1_dims, fc2_dims, first_block=False):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadedAttention(
            input_dims,
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

        self.norm_2 = nn.LayerNorm(encoding_dims)

    def forward(self, inputs):
        mha_output, skip_val = self.multi_headed_attention.forward(inputs)
        add_and_norm = self.norm_1(skip_val + mha_output)
        ff_output = self.feed_forward(add_and_norm)
        output = self.norm_2(add_and_norm + ff_output)
        ## output dims (N, 1 + (input_dims[-2] * input_dims[-1]), encoding_dim)
        return output

class TransformerNetwork(nn.Module):
    def __init__(self, input_dims, encoding_dims, n_heads, \
        fc1_dims, fc2_dims, n_encoder_blocks):
        super(TransformerNetwork, self).__init__()
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(input_dims, encoding_dims, n_heads, fc1_dims, \
                fc2_dims, first_block=True)] + \
            [TransformerEncoder(input_dims, encoding_dims, n_heads, fc1_dims, \
                fc2_dims) for _ in range(n_encoder_blocks-1)]
        )
        self.network = nn.Sequential(
            *self.encoder_blocks,
            nn.LayerNorm(encoding_dims)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)