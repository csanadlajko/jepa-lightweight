import torch.nn as nn
import torch
from ..utils.masking import apply_mask
from ..utils.pos_encoding import sinusoidal_pos_embedding2d
from .vit import TransformerEncoder

class ViTPredictor(nn.Module):

    def __init__(
            self, 
            num_patches,
            device, 
            embed_dim=256, 
            pred_dim=None, 
            depth=6,
            num_heads=8, 
            drop_rate=0.1,
            num_classes=10, 
            tokenizer=None, 
            text_encoder=None
        ):
        super().__init__()
        if pred_dim is None:
            pred_dim = embed_dim

        self.predictor_embed = nn.Linear(embed_dim, pred_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim), requires_grad=True) # learnable parameters to predict masked region

        self.pred_pos_embed = sinusoidal_pos_embedding2d(num_patches, embed_dim, device)

        self.pred_blocks = nn.Sequential(*[
            TransformerEncoder(
                num_heads=num_heads,
                embed_dim=pred_dim,
                mlp_dim=256,
                drop=drop_rate
            )
            for _ in range(depth)
        ])

        self.label_to_embed = nn.Sequential(
            nn.Linear(384, pred_dim), ## depends on the embedding size of the text encoder
            nn.LayerNorm(pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, embed_dim)
        )

        self.cls_fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.cls_fc2 = nn.Linear(embed_dim // 2, num_classes)

        nn.init.xavier_uniform_(self.cls_fc1.weight)
        nn.init.zeros_(self.cls_fc1.bias)
        nn.init.xavier_uniform_(self.cls_fc2.weight)
        nn.init.zeros_(self.cls_fc2.bias)

        self.device = device

        self.cls_head = nn.Sequential(
            self.cls_fc1,
            nn.GELU(),
            nn.Dropout(drop_rate),
            self.cls_fc2
        )

        self.post_pred_mhsa = nn.MultiheadAttention(pred_dim, num_heads, drop_rate, batch_first=True)

        self.predictor_norm = nn.LayerNorm(pred_dim)
        self.predictor_proj = nn.Linear(pred_dim, embed_dim) # back to encoder dimension

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def forward(self, x, context_mask, target_mask, labels: torch.Tensor, multimodal: bool, return_cls_only = False, cell_mask=False):
        ## x includes cls token on index 0

        B = x.size(0)
        
        x = self.predictor_embed(x)
        

        target_positions = apply_mask(self.pred_pos_embed.repeat(B, 1, 1), target_mask, predictor=True)

        num_target_tokens = target_positions.size(1)
        mask_tokens = self.mask_token.repeat(target_positions.size(0), num_target_tokens, 1)
        mask_tokens = mask_tokens + target_positions
        
        context_tokens_repeated = x.repeat(target_positions.size(0) // x.size(0), 1, 1)
        
        x = torch.cat([context_tokens_repeated, mask_tokens], dim=1) # full predicted image with cls token on index 0
        
        for block in self.pred_blocks:
            x = block(x)
        
        x = self.predictor_norm(x)
        
        context_length = context_tokens_repeated.size(1)

        predicted_tokens = x[:, context_length:]
        
        predicted_tokens = self.predictor_proj(predicted_tokens)

        # only enter if model is ran in multimodal mode
        if multimodal: 
            
            label_list = [f"a photo of class: {label}" for label in labels]

            label_tokens: dict[str, torch.Tensor] = self.tokenizer(label_list, return_tensors='pt', padding=True)

            label_tokens = {k: v.to(self.device) for k, v in label_tokens.items()}
            
            enc_labels = self.text_encoder(**label_tokens, output_hidden_states=True)

            enc_labels = enc_labels.hidden_states[-1]

            num_masks = predicted_tokens.size(0) // B

            enc_labels = enc_labels.unsqueeze(1).expand(-1, num_masks, -1, -1)
            enc_labels = enc_labels.reshape(-1, enc_labels.size(2), enc_labels.size(3))

            enc_labels = enc_labels.to(self.device)

            enc_labels = self.label_to_embed(enc_labels)

            pred_attended, _ = self.post_pred_mhsa(predicted_tokens, enc_labels, enc_labels)

            predicted_tokens = predicted_tokens + pred_attended

            if return_cls_only:
                # full_img = torch.cat([x[:context_length], predicted_tokens], dim=1) ## create new total image embedding with finetuned target predictions -> not neccesary
                # we only use the multimodal approach to the predicted target tokens, these do not affect the cls
                # cls only learns the finetuned embedding by the multimodal learning iterations
                cls_token = x[:, 0, :] # acquire cls token representing predicted image
                return self.cls_head(cls_token)
        
        return predicted_tokens

