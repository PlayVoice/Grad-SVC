import math
import torch

from grad.base import BaseModule
from grad.encoder import TextEncoder
from grad.diffusion import Diffusion
from grad.utils import f0_to_coarse, rand_ids_segments, slice_segments


class GradTTS(BaseModule):
    def __init__(self, n_mels, n_vecs, n_pits, n_spks, n_embs, 
                 n_enc_channels, filter_channels, 
                 dec_dim, beta_min, beta_max, pe_scale):
        super(GradTTS, self).__init__()
        # common
        self.n_mels = n_mels
        self.n_vecs = n_vecs
        self.n_spks = n_spks
        self.n_embs = n_embs
        # encoder
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        # decoder
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.pit_emb = torch.nn.Embedding(n_pits, n_embs)
        self.spk_emb = torch.nn.Linear(n_spks, n_embs)
        self.encoder = TextEncoder(n_vecs,
                                   n_mels,
                                   n_embs,
                                   n_enc_channels,
                                   filter_channels)
        self.decoder = Diffusion(n_mels, dec_dim, n_embs, beta_min, beta_max, pe_scale)

    @torch.no_grad()
    def forward(self, lengths, vec, pit, spk, n_timesteps, temperature=1.0, stoc=False):
        """
        Generates mel-spectrogram from vec. Returns:
            1. encoder outputs
            2. decoder outputs

        Args:
            lengths (torch.Tensor): lengths of texts in batch.
            vec (torch.Tensor): batch of speech vec
            pit (torch.Tensor): batch of speech pit
            spk (torch.Tensor): batch of speaker
            
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
        """
        lengths, vec, pit, spk = self.relocate_input([lengths, vec, pit, spk])

        # Get pitch embedding
        pit = self.pit_emb(f0_to_coarse(pit))

        # Get speaker embedding
        spk = self.spk_emb(spk)

        # Transpose
        vec = torch.transpose(vec, 1, -1)
        pit = torch.transpose(pit, 1, -1)

        # Get encoder_outputs `mu_x`
        mu_x, mask_x = self.encoder(lengths, vec, pit, spk)
        encoder_outputs = mu_x

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_x + torch.randn_like(mu_x, device=mu_x.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(spk, z, mask_x, mu_x, n_timesteps, stoc)

        return encoder_outputs, decoder_outputs

    def compute_loss(self, lengths, vec, pit, spk, mel, out_size):
        """
        Computes 2 losses:
            1. prior loss: loss between mel-spectrogram and encoder outputs.
            2. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            lengths (torch.Tensor): lengths of texts in batch.
            vec (torch.Tensor): batch of speech vec
            pit (torch.Tensor): batch of speech pit
            spk (torch.Tensor): batch of speaker
            mel (torch.Tensor): batch of corresponding mel-spectrogram

            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        lengths, vec, pit, spk, mel = self.relocate_input([lengths, vec, pit, spk, mel])

        # Get pitch embedding
        pit = self.pit_emb(f0_to_coarse(pit))

        # Get speaker embedding
        spk = self.spk_emb(spk)

        # Transpose
        vec = torch.transpose(vec, 1, -1)
        pit = torch.transpose(pit, 1, -1)

        # Get encoder_outputs `mu_x`
        mu_x, mask_x = self.encoder(lengths, vec, pit, spk)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            ids = rand_ids_segments(lengths, out_size)
            mel = slice_segments(mel, ids, out_size)

            mask_y = slice_segments(mask_x, ids, out_size)
            mu_y = slice_segments(mu_x, ids, out_size)

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(spk, mel, mask_y, mu_y)

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((mel - mu_y) ** 2 + math.log(2 * math.pi)) * mask_y)
        prior_loss = prior_loss / (torch.sum(mask_y) * self.n_mels)

        return prior_loss, diff_loss
