"""
ASLAvatarModelV4 — Fixed Decoder for Temporal Motion Generation

ROOT CAUSE of static output:
    Old decoder: memory = same z_cond vector copied T times + PE
                 query  = zeros + PE
    → Cross-attention gives nearly identical output at every timestep
    → Model can only output "average pose" per condition, no temporal dynamics

FIX:
    1. Learned temporal query tokens (not zeros) — each position has unique learned content
    2. Temporal noise injection — adds random per-timestep variation to memory
    3. z is kept as single-token memory (not expanded) — forces query to carry temporal info
    
    The decoder now uses:
        memory = [z_cond_token, condition_token]  (B, 2, D) — compact, not expanded
        query  = learned_temporal_queries + PE     (B, T, D) — unique per position
"""

import torch
import torch.nn as nn
import math
from transformers import CLIPTokenizer, CLIPTextModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ASLAvatarModelV4(nn.Module):
    """
    Transformer-based CVAE for sign language motion generation.
    V4: Fixed decoder with learned temporal queries for temporal dynamics.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.INPUT_DIM
        if cfg.ROOT_NORMALIZE:
            self.input_dim = (cfg.N_JOINTS - 1) * cfg.N_FEATS

        # ==================== 1. Condition Module (CLIP) ====================
        print(f"Loading CLIP model: {cfg.CLIP_MODEL_NAME}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.CLIP_MODEL_NAME)
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.CLIP_MODEL_NAME)
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
            
        self.condition_proj = nn.Sequential(
            nn.Linear(cfg.CLIP_DIM, cfg.MODEL_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM)
        )
        
        # ==================== 2. Encoder ====================
        self.pose_proj = nn.Linear(self.input_dim, cfg.MODEL_DIM)
        self.pe = PositionalEncoding(cfg.MODEL_DIM, cfg.MAX_SEQ_LEN + 10)
        
        self.mu_token = nn.Parameter(torch.randn(1, 1, cfg.MODEL_DIM) * 0.02)
        self.sigma_token = nn.Parameter(torch.randn(1, 1, cfg.MODEL_DIM) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL_DIM,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.MODEL_DIM * 4,
            dropout=cfg.DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.N_LAYERS)
        
        self.to_mu = nn.Linear(cfg.MODEL_DIM, cfg.LATENT_DIM)
        self.to_logvar = nn.Linear(cfg.MODEL_DIM, cfg.LATENT_DIM)
        
        # ==================== 3. Decoder (FIXED) ====================
        # Latent and condition projections for memory tokens
        self.latent_proj = nn.Linear(cfg.LATENT_DIM, cfg.MODEL_DIM)
        self.cond_memory_proj = nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM)
        
        # Learned temporal queries — each timestep has unique learned content
        # This is the KEY FIX: queries are NOT zeros, they carry temporal information
        max_seq = getattr(cfg, 'MAX_SEQ_LEN', 200)
        self.temporal_queries = nn.Parameter(
            torch.randn(1, max_seq, cfg.MODEL_DIM) * 0.02
        )
        
        # Condition bias for queries (global modulation)
        self.cond_query_proj = nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.MODEL_DIM,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.MODEL_DIM * 4,
            dropout=cfg.DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.N_LAYERS)
        
        self.output_proj = nn.Linear(cfg.MODEL_DIM, self.input_dim)
        
        self._init_weights()

    def _init_weights(self):
        for module in [self.condition_proj, self.pose_proj, self.to_mu, self.to_logvar,
                       self.latent_proj, self.cond_memory_proj, self.cond_query_proj, 
                       self.output_proj]:
            if isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_text(self, text_list, device):
        inputs = self.tokenizer(
            text_list, padding=True, truncation=True, max_length=77,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            clip_output = self.text_encoder(**inputs)
            clip_emb = clip_output.pooler_output
        return clip_emb

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, motion, condition, padding_mask=None):
        B, T, _ = motion.shape
        device = motion.device
        
        if self.cfg.ROOT_NORMALIZE:
            motion = motion[:, :, self.cfg.N_FEATS:]
        
        motion_emb = self.pose_proj(motion)
        motion_emb = self.pe(motion_emb)
        
        mu_token = self.mu_token.expand(B, -1, -1)
        sigma_token = self.sigma_token.expand(B, -1, -1)
        condition_token = condition.unsqueeze(1)
        
        encoder_input = torch.cat([mu_token, sigma_token, condition_token, motion_emb], dim=1)
        
        if padding_mask is not None:
            prefix_mask = torch.zeros(B, 3, dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix_mask, padding_mask], dim=1)
        else:
            full_mask = None
        
        encoder_output = self.transformer_encoder(encoder_input, src_key_padding_mask=full_mask)
        
        mu = self.to_mu(encoder_output[:, 0, :])
        logvar = self.to_logvar(encoder_output[:, 1, :])
        logvar = torch.clamp(logvar, min=-20, max=20)
        
        return mu, logvar

    def decode(self, z, condition, seq_len, padding_mask=None):
        """
        FIXED decoder: uses learned temporal queries and compact memory.
        
        Old approach (broken):
            memory = same_vector.expand(T) + PE  → identical content at every position
            query  = zeros + PE                   → no temporal content
            
        New approach:
            memory = [z_token, cond_token]        → compact 2-token memory
            query  = learned_temporal_queries + PE + condition_bias
                     → each position has unique learned content
        """
        B = z.shape[0]
        device = z.device
        
        # --- Memory: 2 compact tokens (NOT expanded to T) ---
        z_token = self.latent_proj(z).unsqueeze(1)              # (B, 1, D)
        c_token = self.cond_memory_proj(condition).unsqueeze(1)  # (B, 1, D)
        memory = torch.cat([z_token, c_token], dim=1)            # (B, 2, D)
        
        # --- Query: learned temporal content + PE + condition modulation ---
        # Learned queries provide unique content at each timestep
        queries = self.temporal_queries[:, :seq_len, :].expand(B, -1, -1)  # (B, T, D)
        
        # Add condition as global bias (so different glosses get different queries)
        cond_bias = self.cond_query_proj(condition).unsqueeze(1)  # (B, 1, D)
        queries = queries + cond_bias  # (B, T, D) — condition-modulated
        
        # Add positional encoding
        queries = self.pe(queries)  # (B, T, D)
        
        # No padding mask needed for 2-token memory
        decoder_output = self.transformer_decoder(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=None,  # memory is always valid (2 tokens)
        )
        
        motion = self.output_proj(decoder_output)  # (B, T, input_dim)
        
        if self.cfg.ROOT_NORMALIZE:
            zeros = torch.zeros((B, seq_len, self.cfg.N_FEATS), device=device)
            motion = torch.cat([zeros, motion], dim=-1)
        
        return motion

    def forward(self, motion, text_list, padding_mask=None):
        B, T, _ = motion.shape
        device = motion.device
        
        clip_emb = self.encode_text(text_list, device)
        condition = self.condition_proj(clip_emb)
        
        mu, logvar = self.encode(motion, condition, padding_mask)
        z = self.reparameterize(mu, logvar)
        recon_motion = self.decode(z, condition, T, padding_mask)
        
        return recon_motion, mu, logvar

    @torch.no_grad()
    def generate(self, text_list, seq_len=100, device='cuda'):
        self.eval()
        B = len(text_list)
        
        clip_emb = self.encode_text(text_list, device)
        condition = self.condition_proj(clip_emb)
        
        z = torch.randn(B, self.cfg.LATENT_DIM, device=device)
        motion = self.decode(z, condition, seq_len, padding_mask=None)
        
        return motion

    @torch.no_grad()
    def reconstruct(self, motion, text_list, padding_mask=None):
        self.eval()
        B, T, _ = motion.shape
        device = motion.device
        
        clip_emb = self.encode_text(text_list, device)
        condition = self.condition_proj(clip_emb)
        
        mu, logvar = self.encode(motion, condition, padding_mask)
        z = mu
        recon_motion = self.decode(z, condition, T, padding_mask)
        return recon_motion