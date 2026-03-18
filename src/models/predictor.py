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

    def forward(self, x, context_mask, target_mask, labels: torch.Tensor, multimodal: bool, return_cls_only = False, cell_mask=False, ctx_attn_mask=None):
        ## x includes cls token on index 0

        B = x.size(0)
        
        x = self.predictor_embed(x)
        
        # get target positions with padding [B, N, D] where N is padded
        target_positions, pred_target_attn = apply_mask(self.pred_pos_embed.repeat(B, 1, 1), target_mask, predictor=True, use_padding=cell_mask)
        num_target_tokens = target_positions.size(1)
        # in case of padding, the learnable mask tokens are containing the paddings...
        mask_tokens = self.mask_token.repeat(target_positions.size(0), num_target_tokens, 1)
        mask_tokens = mask_tokens + target_positions
        
        context_tokens_repeated = x.repeat(target_positions.size(0) // x.size(0), 1, 1)
        
        x = torch.cat([context_tokens_repeated, mask_tokens], dim=1) # full predicted image with cls token on index 0

        if cell_mask is True and ctx_attn_mask is not None:
            cc_attn = torch.cat([ctx_attn_mask, pred_target_attn], dim=1)
            assert cc_attn.shape[1] == x.shape[1]
        else: cc_attn = None

        for block in self.pred_blocks:
            x = block(x, cc_attn)
        
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

class CellTypePredictor(nn.Module):

    def __init__(self, num_classes=8, embed_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.pred_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, target_patch_indices: list[list[torch.Tensor]], gt_patch_classes: torch.Tensor, loss_fn):
        """
        :x predicted target patch embeddings with shape [B, N, D] where N is the (same) padded number of target patches in the batch
        :target_patch_indices list of length B containing lists of one index tensor each
        :gt_patch_classes ground-truth class labels for the target patches, shape [B, N] where N is the maximum number of patches in an image
        """

        total_losses = []
        total_samples = 0
        total_correct = 0

        # for every image in the batch
        for b_idx, batch in enumerate(target_patch_indices):
            # for every target patch index in an image
            patch_indices = batch[0].to(x.device)
            
            gt_class = gt_patch_classes[b_idx].index_select(dim=0, index=patch_indices)
            # only select the non padded indices from tensor x
            pred_classes = self.pred_layer(x[b_idx, :patch_indices.shape[0], :])
            loss = loss_fn(pred_classes, gt_class)
            pred_labels = torch.argmax(pred_classes, dim=1)

            correct = (pred_labels == gt_class).sum().item()

            total_correct += correct
            total_samples += gt_class.numel()
            total_losses.append(loss)

        avg_loss = sum(total_losses) / len(total_losses)
        avg_accuracy = (total_correct / total_samples) * 100

        return avg_loss, avg_accuracy
