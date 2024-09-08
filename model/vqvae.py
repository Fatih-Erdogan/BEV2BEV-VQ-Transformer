import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.vqvae_helper_modules import SamePadConv3d, SamePadConvTranspose3d, AttentionResidualBlock
from collections import OrderedDict


class VQVAE3D(nn.Module):
    def __init__(self, encoder, decoder, codebook):
        super(VQVAE3D, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = codebook

    def encode_code(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            # bs * t * h' * w'
            return indices

    def decode_code(self, latents):
        # latents: bs * t * h' * w'
        with torch.no_grad():
            # move channel to front
            latents = self.codebook.embedding(latents).permute(0, 4, 1, 2, 3).contiguous()
            # bs * c * t * h' * w'
            # moves channel to end
            return self.decoder(latents).permute(0, 2, 3, 4, 1).cpu().numpy()  # bs * t * h * w * c

    def forward(self, x):
        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decoder(e_st)

        diff1 = torch.mean((z - e.detach()) ** 2)  # solely for the encoder update (commitment loss)
        diff2 = torch.mean((e - z.detach()) ** 2)  # solely for the codebook update (quantization/codebook loss)
        return x_tilde, diff1 + diff2

    def loss(self, x):
        x_tilde, diff = self(x)
        # notice that the MSE is calculated as mean, not the sum(all) / batch_size
        recon_loss = F.mse_loss(x_tilde, x)  # for encoder and the decoder, but not for the codebook
        loss = recon_loss + diff
        return OrderedDict(loss=loss, recon_loss=recon_loss, vq_loss=diff)


# modules for 128 * 128 input
class VQ3DEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, time_window, downsample, n_res_layers, bias=True):
        super(VQ3DEncoder, self).__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(downsample):
            cur_in_channels = in_channels if i == 0 else out_channels
            time_kernel = min(time_window, 4)
            conv_block = SamePadConv3d(cur_in_channels,
                                       out_channels,
                                       kernel_size=(time_kernel, 4, 4),
                                       stride=(1, 2, 2),
                                       bias=bias)
            self.conv_blocks.append(conv_block)

        self.last_conv = SamePadConv3d(out_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       bias=bias)

        # bias false for convs in res_blocks
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(out_channels)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [bs * in_c * t * h * w]
        h = x
        for conv in self.conv_blocks:
            h = F.relu(conv(h))
        h = self.last_conv(h)
        h = self.res_stack(h)
        # returns [bs * out_c * t * h * w]
        return h


class VQ3DDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, time_window, upsample, n_res_layers, bias=True):
        # for ex: in channels 256, out channels 3
        super(VQ3DDecoder, self).__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(in_channels)
              for _ in range(n_res_layers)],
            nn.BatchNorm3d(in_channels),
            nn.ReLU()
        )

        self.deconv_blocks = nn.ModuleList()
        for i in range(upsample):
            cur_out_channels = in_channels if i != (upsample - 1) else out_channels
            time_kernel = min(time_window, 4)
            deconv_block = SamePadConvTranspose3d(in_channels,
                                                  cur_out_channels,
                                                  kernel_size=(time_kernel, 4, 4),
                                                  stride=(1, 2, 2),
                                                  bias=bias)
            self.deconv_blocks.append(deconv_block)

    def forward(self, x):
        h = self.res_stack(x)
        for i, deconv in enumerate(self.deconv_blocks):
            h = deconv(h)
            if i < len(self.deconv_blocks) - 1:
                h = F.relu(h)
        return h


# as not diverse data and large dataset, we use the normal codebook loss version, not EMA
class CodeBook(nn.Module):
    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=size, embedding_dim=code_dim)
        init.uniform_(self.embedding.weight.data, -1.0/size, 1.0/size)

    def forward(self, z):
        # You should return the quantized latent, the straight-through estimator and the
        # indices after sorting by the distance, in the [B,H,W] shape
        bs, latent_dim, t, h, w = z.shape
        z_reshaped = z.permute(0, 2, 3, 4, 1).reshape(-1, latent_dim)  # becomes [(bs * t * h * w), latent_dim]

        # embeddings become [num_embed, (bs * t * h * w), latent_dim]
        # dif_vecs = self.embedding.weight.unsqueeze(1).repeat(1, bs * t* h * w, 1) - z_reshaped
        # instead of the upper one, use broadcasting by removing the repeat
        dif_vecs = self.embedding.weight.unsqueeze(1) - z_reshaped
        dif_norms = torch.norm(dif_vecs, dim=-1)  # ends up [num_embed, (bs * t *  h * w)]
        dif_norms = dif_norms.view(-1, bs, t, h, w).permute(1, 2, 3, 4, 0)
        encoding_indices = torch.argmin(dif_norms, dim=-1)  # ends up [bs, t, h, w] containing min indices
        quantized = self.embedding(encoding_indices).permute(0, 4, 1, 2, 3)  # bring the latent dim to channel
        return quantized, z + (quantized - z).detach(), encoding_indices

