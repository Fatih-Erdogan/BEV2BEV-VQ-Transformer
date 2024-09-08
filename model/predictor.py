import torch
import torch.nn as nn
import math
import numpy as np

from model.vqvae import CodeBook
from model.attention import LayerNorm as CondLayerNorm


class ConcatConditionedTransformer(nn.Module):
    def __init__(self, vocab_size, hid_dim, num_blocks, num_heads, attn_drop, seq_len, positional_type="sinusoidal",
                 action_condition=None,  # this one will be ignored, assume always True
                 use_vq_embeddings=False, codebook=None,
                 first_action_only=False, action_rnn_layers=None,
                 num_cond_bins=None, total_cond_dim=None, dim_per_action=None):
        super(ConcatConditionedTransformer, self).__init__()
        assert action_condition, "ConcatConditionedTransformer requires action condition."
        assert (hid_dim + total_cond_dim) % num_heads == 0
        assert total_cond_dim % ((hid_dim + total_cond_dim) // num_heads) == 0, "total_cond_dim must be divisible by " \
                                                                                "the dimensionality of a head"

        # embeddings
        if use_vq_embeddings:
            assert codebook is not None, "use_vq_embeddings=True but no codebook provided!"
            assert isinstance(codebook, CodeBook), "codebook must be an instance of CodeBook class"

            self.embeddings = codebook.embedding
            self.embedding_projection = nn.Linear(codebook.embedding.embedding_dim, hid_dim)
            for param in self.embeddings.parameters():
                param.requires_grad = False
        else:
            self.embeddings = nn.Embedding(vocab_size, hid_dim)
            self.embedding_projection = nn.Identity()

        # positional encodings
        self.seq_len = seq_len
        self.hid_dim = hid_dim + total_cond_dim
        self.positional_enc = self.init_positionals(positional_type)

        # attention blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.hid_dim, num_heads, attn_drop, action_condition=False, cond_dim=None) for _ in
            range(num_blocks)
        ])

        # handle actions
        self.action_embeddings = nn.Embedding(num_embeddings=num_cond_bins, embedding_dim=dim_per_action)
        self.first_action_only = first_action_only
        if not first_action_only:
            self.action_rnn = ActionGRU(total_cond_dim, 2 * total_cond_dim, action_rnn_layers)
        else:
            self.action_rnn = nn.Identity()

        # layer norm
        self.layer_norm = CondLayerNorm(self.hid_dim, class_cond_dim=None)

        # out layer
        self.out_layer = nn.Linear(self.hid_dim, vocab_size)

    def forward(self, x, action):
        # x: bs * seq_len (encoded indices), seq_len is the number of quantized vectors of t frame
        # action: bs * t * n or bs * n (action indices)

        x = self.embeddings(x)
        x = self.embedding_projection(x)  # bs * seq_len * hid_dim

        # get the action embeddings:
        action = self.aggregate_actions(action)  # bs * total_cond_dim
        action = action.unsqueeze(1)             # bs * 1 * total_cond_dim (for broadcast)
        action = action.expand(-1, self.seq_len, -1)

        # concat the actions to the sequence tokens
        x = torch.cat((x, action), dim=-1)      # bs * seq_len * (hid_dim + total_cond_dim)

        # apply the pos enc after the projection
        x = x + self.positional_enc[:self.seq_len].unsqueeze(0)

        for block in self.blocks:
            x = block(x, action=None)

        x = self.layer_norm(x, cond=None)
        logits = self.out_layer(x)

        # bs * seq_len * vocab_size
        return logits

    def aggregate_actions(self, actions):
        if self.first_action_only:
            # actions: bs * n
            bs, _ = actions.shape
            action_vecs = self.action_embeddings(actions)  # bs * n * dim_per_action
            action_vecs = action_vecs.view(bs, -1)  # bs * (n * dim_per_action) dim_per_action's are concatenated
        else:
            # actions: bs * t * n
            bs, t, n = actions.shape
            action_vecs = self.action_embeddings(actions)  # bs * t * n * dim_per_action
            action_vecs = action_vecs.view(bs, t,
                                           -1)  # bs * t * (n * dim_per_action) dim_per_action's are concatenated

        # it should return bs * total_cond_dim (total_cond_dim = n * dim_per_action)
        # if first_action_only -> Identity, else: GRU
        action_vecs = self.action_rnn(action_vecs)

        return action_vecs

    def init_positionals(self, positional_type):
        seq_len = self.seq_len
        hid_dim = self.hid_dim
        if positional_type == 'sinusoidal':
            position = torch.arange(0, seq_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hid_dim, 2) * -(math.log(10000.0) / hid_dim))
            pe = torch.zeros(seq_len, hid_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = nn.Parameter(pe, requires_grad=False)
        elif positional_type == 'learnable':
            pe = nn.Parameter(torch.randn(seq_len, hid_dim))
        else:
            assert False, f"Type {positional_type} positional encodings not implemented"
        return pe


class ActionConditionedTransformer(nn.Module):
    def __init__(self, vocab_size, hid_dim, num_blocks, num_heads, attn_drop, seq_len, positional_type="sinusoidal",
                 action_condition=False, total_cond_dim=None, dim_per_action=None, num_cond_bins=None,
                 first_action_only=True, action_rnn_layers=None,
                 use_vq_embeddings=False, codebook=None):
        super(ActionConditionedTransformer, self).__init__()

        # embeddings
        if use_vq_embeddings:
            assert codebook is not None, "use_vq_embeddings=True but no codebook provided!"
            assert isinstance(codebook, CodeBook), "codebook must be an instance of CodeBook class"

            self.embeddings = codebook.embedding
            self.embedding_projection = nn.Linear(codebook.embedding.embedding_dim, hid_dim)
            for param in self.embeddings.parameters():
                param.requires_grad = False
        else:
            self.embeddings = nn.Embedding(vocab_size, hid_dim)
            self.embedding_projection = nn.Identity()

        # positional encodings
        self.seq_len = seq_len
        self.hid_dim = hid_dim
        self.positional_enc = self.init_positionals(positional_type)

        # attention blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hid_dim, num_heads, attn_drop, action_condition, total_cond_dim) for _ in range(num_blocks)
        ])

        self.action_conditioned = action_condition
        # for action aggregation
        self.first_action_only = first_action_only
        if not first_action_only:
            self.action_rnn = ActionGRU(total_cond_dim, 2 * total_cond_dim, action_rnn_layers)
        else:
            self.action_rnn = nn.Identity()

        # layer norm
        if action_condition:
            assert total_cond_dim is not None and num_cond_bins is not None and dim_per_action is not None
            self.action_embeddings = nn.Embedding(num_embeddings=num_cond_bins, embedding_dim=dim_per_action)
            self.layer_norm = CondLayerNorm(hid_dim, class_cond_dim=total_cond_dim)
        else:
            self.layer_norm = CondLayerNorm(hid_dim, class_cond_dim=None)

        # out layer
        self.out_layer = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, action=None):
        # x: bs * seq_len (encoded indices), seq_len is the number of quantized vectors of t frame
        # action: bs * t * n or bs * n (action indices)

        x = self.embeddings(x)
        x = self.embedding_projection(x)  # bs * seq_len * hid_dim
        # apply the pos enc after the projection
        x = x + self.positional_enc[:self.seq_len].unsqueeze(0)

        action = self.aggregate_actions(action)
        for block in self.blocks:
            x = block(x, action)

        x = self.layer_norm(x, action)
        logits = self.out_layer(x)

        # bs * seq_len * vocab_size
        return logits

    def aggregate_actions(self, actions):
        if self.first_action_only:
            # actions: bs * n
            bs, _ = actions.shape
            action_vecs = self.action_embeddings(actions)  # bs * n * dim_per_action
            action_vecs = action_vecs.view(bs, -1)  # bs * (n * dim_per_action) dim_per_action's are concatenated
        else:
            # actions: bs * t * n
            bs, t, n = actions.shape
            action_vecs = self.action_embeddings(actions)  # bs * t * n * dim_per_action
            action_vecs = action_vecs.view(bs, t, -1)  # bs * t * (n * dim_per_action) dim_per_action's are concatenated

        # it should return bs * total_cond_dim (total_cond_dim = n * dim_per_action)
        # if first_action_only -> Identity, else: GRU
        action_vecs = self.action_rnn(action_vecs)

        return action_vecs

    def init_positionals(self, positional_type):
        seq_len = self.seq_len
        hid_dim = self.hid_dim
        if positional_type == 'sinusoidal':
            position = torch.arange(0, seq_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hid_dim, 2) * -(math.log(10000.0) / hid_dim))
            pe = torch.zeros(seq_len, hid_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = nn.Parameter(pe, requires_grad=False)
        elif positional_type == 'learnable':
            pe = nn.Parameter(torch.randn(seq_len, hid_dim))
        else:
            assert False, f"Type {positional_type} positional encodings not implemented"
        return pe


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, action_condition=False, cond_dim=None):
        super(TransformerBlock, self).__init__()
        if action_condition: assert cond_dim is not None
        self.layer_norm1 = CondLayerNorm(embd_dim=dim, class_cond_dim=cond_dim)
        self.mh_attention = MultiHeadAttention(d_model=dim, heads=num_heads, dropout=dropout)
        self.layer_norm2 = CondLayerNorm(embd_dim=dim, class_cond_dim=cond_dim)
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim * 4),
            nn.GELU(),
            nn.Linear(in_features=dim * 4, out_features=dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, action=None):
        # add happens at the end of prev layer (see the line before return)
        h = self.layer_norm1(x, action)
        h = self.mh_attention(h, mask=None)
        x = x + h

        h = self.layer_norm2(x, action)
        h = self.fc_block(h)
        x = x + h

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        assert d_model % heads == 0

        self.num_heads = heads
        self.heads = nn.ModuleList()

        for i in range(heads):
            self.heads.append(SingleHeadAttention(d_model // heads, dropout))

        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        """
        # x becomes bs, seq_len, num_heads, dim // num_heads
        x = x.view(x.shape[0], x.shape[1], self.num_heads, x.shape[2] // self.num_heads)
        heads_list = list()
        for idx, head in enumerate(self.heads):
            heads_list.append(head(x[:, :, idx, :], x[:, :, idx, :], x[:, :, idx, :], mask))
        out = torch.cat(heads_list, dim=2)

        out = self.proj(out)
        out = self.drop(out)

        # (batch_size, seq_len, d_model)
        return out


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, dropout=None):
        super().__init__()
        self.d_model = d_model
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, q, v, mask=None):
        """
        x: (batch_size, seq_len, d_model // num_heads)
        mask: (seq_len, seq_len) for training and inference
        """
        key = self.key(k)
        query = self.query(q)
        value = self.value(v)

        # logits output shape (bs, seq_len, seq_len)
        logits = torch.matmul(query, key.transpose(1, 2)) / np.sqrt(self.d_model)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -np.inf)
        probs = torch.softmax(logits, dim=2)
        probs = self.dropout(probs)

        # out: (batch_size, seq_len, d_model // num_heads)
        return torch.matmul(probs, value)


class ActionGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers):
        super(ActionGRU, self).__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=True)
        self.projection = nn.Linear(hid_dim, in_dim, bias=False)

    def forward(self, x):
        _, h_n = self.gru(x)  # h_n -> num_layers * bs * hid_dim
        h_n = self.projection(h_n[-1])
        return h_n
