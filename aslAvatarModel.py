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
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class ASLAvatarModel(nn.Module):
    """
    Transformer-based CVAE for sign language motion generation.
    Based on SignAvatar architecture with modifications for skeleton input.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # ==================== 1. Condition Module (CLIP) ====================
        print(f"Loading CLIP model: {cfg.CLIP_MODEL_NAME}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.CLIP_MODEL_NAME)
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.CLIP_MODEL_NAME)
        
        # Freeze CLIP
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
            
        # CLIP -> Model dim projection (3-layer MLP)
        self.condition_proj = nn.Sequential(
            nn.Linear(cfg.CLIP_DIM, cfg.MODEL_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM)
        )
        
        # ==================== 2. Encoder ====================
        # Pose projection: skeleton coords -> model dim
        self.pose_proj = nn.Linear(cfg.INPUT_DIM, cfg.MODEL_DIM)
        self.pe = PositionalEncoding(cfg.MODEL_DIM, cfg.MAX_SEQ_LEN + 10)
        
        # Learnable distribution tokens
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
        
        # Distribution heads
        self.to_mu = nn.Linear(cfg.MODEL_DIM, cfg.LATENT_DIM)
        self.to_logvar = nn.Linear(cfg.MODEL_DIM, cfg.LATENT_DIM)
        
        # ==================== 3. Decoder ====================
        # Latent + condition fusion
        self.latent_proj = nn.Linear(cfg.LATENT_DIM, cfg.MODEL_DIM)
        self.cond_bias_proj = nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.MODEL_DIM,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.MODEL_DIM * 4,
            dropout=cfg.DROPOUT,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.N_LAYERS)
        
        # Output projection: model dim -> skeleton coords
        self.output_proj = nn.Linear(cfg.MODEL_DIM, cfg.INPUT_DIM)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for non-pretrained modules"""
        for module in [self.condition_proj, self.pose_proj, self.to_mu, self.to_logvar,
                       self.latent_proj, self.cond_bias_proj, self.output_proj]:
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
        """Encode text using frozen CLIP"""
        inputs = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            max_length=77,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            clip_output = self.text_encoder(**inputs)
            # Use pooled output (CLS token representation)
            clip_emb = clip_output.pooler_output  # (B, 512)
        return clip_emb

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Use mean during inference for reconstruction

    def encode(self, motion, condition, padding_mask=None):
        """
        Encoder: motion + condition -> distribution parameters (mu, logvar)
        
        Args:
            motion: (B, T, D_pose) - pose sequence
            condition: (B, D_model) - projected condition embedding
            padding_mask: (B, T) - True where padded
        
        Returns:
            mu, logvar: (B, D_latent) each
        """
        B, T, _ = motion.shape
        device = motion.device
        
        # Project motion to model dim and add PE
        motion_emb = self.pose_proj(motion)  # (B, T, D_model)
        motion_emb = self.pe(motion_emb)
        
        # Expand learnable tokens
        mu_token = self.mu_token.expand(B, -1, -1)      # (B, 1, D_model)
        sigma_token = self.sigma_token.expand(B, -1, -1)  # (B, 1, D_model)
        condition = condition.unsqueeze(1)              # (B, 1, D_model)
        
        # Construct encoder input: [mu_token, sigma_token, condition, motion...]
        encoder_input = torch.cat([mu_token, sigma_token, condition, motion_emb], dim=1)
        
        # Adjust padding mask for prepended tokens
        if padding_mask is not None:
            prefix_mask = torch.zeros(B, 3, dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix_mask, padding_mask], dim=1)
        else:
            full_mask = None
        
        # Transformer encoder
        encoder_output = self.transformer_encoder(encoder_input, src_key_padding_mask=full_mask)
        
        # Extract distribution parameters from token positions
        mu = self.to_mu(encoder_output[:, 0, :])       # (B, D_latent)
        logvar = self.to_logvar(encoder_output[:, 1, :])  # (B, D_latent)
        
        return mu, logvar

    def decode(self, z, condition, seq_len, padding_mask=None):
        """
        Decoder: z + condition -> reconstructed motion
        
        Args:
            z: (B, D_latent) - sampled latent
            condition: (B, D_model) - projected condition embedding
            seq_len: int - target sequence length
            padding_mask: (B, T) - True where padded
        
        Returns:
            motion: (B, T, D_pose)
        """
        B = z.shape[0]
        device = z.device
        
        # Combine latent and condition: z_cond = MLP(z) + MLP(c)
        z_proj = self.latent_proj(z)           # (B, D_model)
        c_bias = self.cond_bias_proj(condition)  # (B, D_model)
        z_cond = z_proj + c_bias               # (B, D_model)
        
        # Expand to sequence: memory for cross-attention
        memory = z_cond.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, D_model)
        
        # Query: pure positional encoding (indicates "which frame")
        query = torch.zeros(B, seq_len, self.cfg.MODEL_DIM, device=device)
        query = self.pe(query)  # (B, T, D_model)
        
        # Transformer decoder: query attends to memory
        decoder_output = self.transformer_decoder(
            tgt=query,
            memory=memory,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask
        )
        
        # Project to pose space
        motion = self.output_proj(decoder_output)  # (B, T, D_pose)
        
        return motion

    def forward(self, motion, text_list, padding_mask=None):
        """
        Full forward pass for training (reconstruction)
        
        Args:
            motion: (B, T, D_pose) - input pose sequence
            text_list: List[str] - gloss names
            padding_mask: (B, T) - True where padded
        
        Returns:
            recon_motion: (B, T, D_pose)
            mu, logvar: (B, D_latent) each
        """
        B, T, _ = motion.shape
        device = motion.device
        
        # Encode condition (text -> CLIP -> projection)
        clip_emb = self.encode_text(text_list, device)  # (B, 512)
        condition = self.condition_proj(clip_emb)        # (B, D_model)
        
        # Encode motion -> distribution
        mu, logvar = self.encode(motion, condition, padding_mask)
        
        # Sample from distribution
        z = self.reparameterize(mu, logvar)  # (B, D_latent)
        
        # Decode to motion
        recon_motion = self.decode(z, condition, T, padding_mask)
        
        return recon_motion, mu, logvar

    @torch.no_grad()
    def generate(self, text_list, seq_len=100, device='cuda'):
        """
        Generation: sample from prior N(0,I) and decode
        
        Args:
            text_list: List[str] - gloss names
            seq_len: int - desired output sequence length
            device: torch device
        
        Returns:
            motion: (B, T, D_pose)
        """
        self.eval()
        B = len(text_list)
        
        # Encode condition
        clip_emb = self.encode_text(text_list, device)
        condition = self.condition_proj(clip_emb)
        
        # Sample from prior N(0, I)
        z = torch.randn(B, self.cfg.LATENT_DIM, device=device)
        
        # Decode
        motion = self.decode(z, condition, seq_len, padding_mask=None)
        
        return motion

    @torch.no_grad()
    def reconstruct(self, motion, text_list, padding_mask=None):
        """
        Reconstruction: encode then decode (using mean, not sampling)
        """
        self.eval()
        B, T, _ = motion.shape
        device = motion.device
        
        clip_emb = self.encode_text(text_list, device)
        condition = self.condition_proj(clip_emb)
        
        mu, logvar = self.encode(motion, condition, padding_mask)
        # Use mean for reconstruction
        z = mu
        
        recon_motion = self.decode(z, condition, T, padding_mask)
        return recon_motion