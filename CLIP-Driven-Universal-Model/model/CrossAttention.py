import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalTransformer(nn.Module):
    def __init__(self, text_emb_dim, img_emb_dim, num_heads):
        super(CrossModalTransformer, self).__init__()
        self.text_emb_dim = text_emb_dim
        self.img_emb_dim = img_emb_dim
        self.num_heads = num_heads
        
        # Text embedding projection layer
        #self.text_projection = nn.Linear(text_emb_dim, img_emb_dim)
        # Image embedding projection layer
        #self.img_projection = nn.Linear(img_emb_dim, text_emb_dim)
        
        # Multi-head attention layers for text and image embeddings
        self.text_self_attention = nn.MultiheadAttention(text_emb_dim, num_heads, batch_first=True)
        self.img_self_attention = nn.MultiheadAttention(img_emb_dim, num_heads, batch_first=True)
        
        # Cross-modal attention layer
        self.cross_modal_attention_text = nn.MultiheadAttention(text_emb_dim, num_heads, batch_first=True)
        self.cross_modal_attention_image = nn.MultiheadAttention(img_emb_dim, num_heads, batch_first=True)
        
        # Layer normalization for all attention layers
        self.text_norm1 = nn.LayerNorm(img_emb_dim)
        self.img_norm1 = nn.LayerNorm(text_emb_dim)
        self.cross_modal_norm1_text = nn.LayerNorm(text_emb_dim)
        self.cross_modal_norm1_image = nn.LayerNorm(img_emb_dim)
        
        # Feedforward layers for all attention layers
        self.text_feedforward = nn.Sequential(
            nn.Linear(img_emb_dim, img_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(img_emb_dim * 4, img_emb_dim)
        )
        self.img_feedforward = nn.Sequential(
            nn.Linear(text_emb_dim, text_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(text_emb_dim * 4, text_emb_dim)
        )
        self.cross_modal_feedforward_text = nn.Sequential(
            nn.Linear(text_emb_dim, text_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(text_emb_dim * 4, text_emb_dim)
        )
        self.cross_modal_feedforward_image = nn.Sequential(
            nn.Linear(img_emb_dim, img_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(img_emb_dim * 4, img_emb_dim)
        )
        
        # Layer normalization for all feedforward layers
        self.text_norm2 = nn.LayerNorm(img_emb_dim)
        self.img_norm2 = nn.LayerNorm(text_emb_dim)
        self.cross_modal_norm2_text = nn.LayerNorm(text_emb_dim)
        self.cross_modal_norm2_image = nn.LayerNorm(img_emb_dim)
    
    def forward(self, text_emb, img_emb):
        # Project text embedding to image embedding space
        #text_emb_proj = self.text_projection(text_emb)
        
        # Self-attention on text and image embeddings
        #text_emb_self, _ = self.text_self_attention(text_emb_proj, text_emb_proj, text_emb_proj)
        #text_emb = self.text_norm1(text_emb_proj + text_emb_self)
        text_emb_self, _ = self.text_self_attention(text_emb, text_emb, text_emb)
        text_emb = self.text_norm1(text_emb + text_emb_self)
        text_emb = self.text_norm2(text_emb + self.text_feedforward(text_emb))
        
        img_emb_self, _ = self.img_self_attention(img_emb, img_emb, img_emb)
        img_emb = self.img_norm1(img_emb + img_emb_self)
        img_emb = self.img_norm2(img_emb + self.img_feedforward(img_emb))
        
        # Cross-modal attention between text and image embeddings
        cross_modal_emb, _ = self.cross_modal_attention_text(text_emb, img_emb, img_emb)
        text_emb = self.cross_modal_norm1_text(text_emb + cross_modal_emb)
        text_emb = self.cross_modal_norm2_text(text_emb + self.cross_modal_feedforward_text(text_emb))

        # Cross-modal attention between text and image embeddings
        cross_modal_emb, _ = self.cross_modal_attention_image(img_emb, text_emb, text_emb)
        img_emb = self.cross_modal_norm1_image(img_emb + cross_modal_emb)
        img_emb = self.cross_modal_norm2_image(img_emb + self.cross_modal_feedforward_image(img_emb))

        return text_emb, img_emb
