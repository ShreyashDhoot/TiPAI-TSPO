#!/usr/bin/env python3
"""
Standalone Inference Script for Adversarial Image Auditor (ResNet101 Backbone)
Supports 5-class safety taxonomy: Safe, Violence, Sexual, Illegal Activity, Disturbing
Usage:
    python3 auditor_inference.py --model checkpoints/complete_auditor_best.pth --image sample.jpg --prompt "sample prompt"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image as PILImage
import os
import json
import argparse
import numpy as np
from typing import List

# =============================================================================
# MODEL ARCHITECTURE (Synced with training)
# =============================================================================

class SimpleTokenizer:
    """Simple word-level tokenizer"""
    def __init__(self, vocab_path=None, max_length=77):
        self.max_length = max_length
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                self.word_to_idx = json.load(f)
            print(f"[+] Loaded vocabulary from {vocab_path} ({len(self.word_to_idx)} words)")
    
    def encode(self, text):
        """Tokenize text to indices"""
        if not text:
            return torch.zeros(self.max_length, dtype=torch.long)
        
        words = str(text).lower().split()
        indices = [self.word_to_idx.get('<SOS>', 2)]
        
        for word in words[:self.max_length-2]:
            idx = self.word_to_idx.get(word, self.word_to_idx.get('<UNK>', 1))
            indices.append(idx)
        
        indices.append(self.word_to_idx.get('<EOS>', 3))
        while len(indices) < self.max_length:
            indices.append(0)
        
        return torch.tensor(indices[:self.max_length], dtype=torch.long)

class SimpleTextEncoder(nn.Module):
    """Word-embedding BiLSTM text encoder"""
    def __init__(self, vocab_size=50000, embed_dim=512, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 512)
        self.norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_tokens):
        padding_mask = (text_tokens == 0)
        embedded = self.dropout(self.embedding(text_tokens))
        out, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        text_features = self.fc(hidden)
        seq_features = self.norm(self.fc(out))
        return text_features, seq_features, padding_mask

class CompleteMultiTaskAuditor(nn.Module):
    """ResNet101 multi-task adversarial image auditor (Inference Version)"""
    def __init__(self, num_classes=5, vocab_size=50000):
        super().__init__()
        resnet = models.resnet101(weights=None) # We'll load weights later
        self.features = nn.Sequential(*list(resnet.children())[:-2]) 

        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size)
        self.adv_head    = nn.Conv2d(2048, 1, kernel_size=1)
        self.class_head  = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.quality_head = nn.Conv2d(2048, 1, kernel_size=1)
        self.object_detection_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        self.image_proj   = nn.Conv2d(2048, 512, kernel_size=1)
        self.query_norm   = nn.LayerNorm(512)
        self.key_norm     = nn.LayerNorm(512)
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        self.img_proj_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256))
        self.txt_proj_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256))
        self.log_temperature = nn.Parameter(torch.tensor([-2.659]))

        self.timestep_embed = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 512)
        )
        self.film_adv  = nn.Linear(512, 2048 * 2)
        self.film_seam = nn.Linear(512, 512 * 2)

        self.relative_adv_head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.seam_feat = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.seam_cls = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x, text_tokens=None, timestep=None, return_features=False):
        B = x.size(0)
        feats = self.features(x)
        global_feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)

        adv_logits = F.adaptive_avg_pool2d(self.adv_head(feats), (1, 1)).flatten(1)
        class_logits = F.adaptive_avg_pool2d(self.class_head(feats), (1, 1)).flatten(1)
        qual_logits = F.adaptive_avg_pool2d(self.quality_head(feats), (1, 1)).flatten(1)
        
        text_features, seq_features, padding_mask = self.text_encoder(text_tokens)
        
        img_feats_proj = self.image_proj(feats)
        Bi, Ci, Hi, Wi = img_feats_proj.shape
        img_seq = self.query_norm(img_feats_proj.view(Bi, Ci, -1).permute(0, 2, 1))
        seq_feat_normed = self.key_norm(seq_features)
        
        attended_img_seq, _ = self.cross_attention(img_seq, seq_feat_normed, seq_feat_normed, key_padding_mask=padding_mask)
        attended_img_feat = attended_img_seq.mean(dim=1)
        
        img_embed = F.normalize(self.img_proj_head(attended_img_feat), dim=-1)
        txt_embed = F.normalize(self.txt_proj_head(text_features), dim=-1)

        ts_feat = self.timestep_embed(timestep)
        gbeta_adv = self.film_adv(ts_feat)
        gamma_adv, beta_adv = gbeta_adv.chunk(2, dim=-1)
        global_feats_mod = (1.0 + gamma_adv) * global_feats + beta_adv
        
        relative_adv_score = torch.sigmoid(self.relative_adv_head(global_feats_mod))
        
        seam_mid = self.seam_feat(feats)
        gamma_seam, beta_seam = self.film_seam(ts_feat).chunk(2, dim=-1)
        seam_mid = (1.0 + gamma_seam[:, :, None, None]) * seam_mid + beta_seam[:, :, None, None]
        seam_quality_score = F.adaptive_avg_pool2d(torch.sigmoid(self.seam_cls(seam_mid)), (1, 1)).flatten(1)

        out = {
            'binary_logits':      adv_logits,
            'class_logits':       class_logits,
            'quality_logits':     qual_logits,
            'img_embed':          img_embed,
            'txt_embed':          txt_embed,
            'seam_quality_score': seam_quality_score,
            'relative_adv_score': relative_adv_score
        }

        if return_features:
            out['adversarial_map'] = torch.sigmoid(self.adv_head(feats))
            out['object_heatmaps'] = torch.sigmoid(self.object_detection_head(feats))
            
        return out

# =============================================================================
# INFERENCE UTILITIES
# =============================================================================

CLASS_NAMES = ['Safe', 'Violence', 'Sexual', 'Illegal Activity', 'Disturbing']

def predict_single(model, tokenizer, image_path, prompt="", return_heatmaps=False, timestep_value=0.0):
    device = next(model.parameters()).device
    
    # Load and transform image
    image = PILImage.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Process text
    text_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
    
    timestep = torch.tensor([[float(timestep_value)]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor, text_tokens=text_tokens, timestep=timestep, return_features=return_heatmaps)
        
    # Process outputs
    binary_prob = torch.sigmoid(outputs['binary_logits']).item()
    class_probs = F.softmax(outputs['class_logits'], dim=1)[0].cpu().numpy()
    rel_adv = outputs['relative_adv_score'].item()
    seam_qual = outputs['seam_quality_score'].item()
    
    # Cosine similarity for faithfulness
    cos_sim = (outputs['img_embed'] @ outputs['txt_embed'].T).item()
    faithfulness = (cos_sim + 1.0) / 2.0  # Normalized to 0-1
    
    res = {
        "is_adversarial": binary_prob > 0.5,
        "adversarial_probability": binary_prob,
        "primary_category": CLASS_NAMES[np.argmax(class_probs)],
        "category_probabilities": {CLASS_NAMES[i]: float(class_probs[i]) for i in range(len(CLASS_NAMES))},
        "relative_adversary_score": rel_adv,
        "seam_quality_assessment": seam_qual,
        "text_faithfulness_score": faithfulness
    }

    if return_heatmaps:
        # Resize heatmaps to original image size
        adv_map = F.interpolate(outputs['adversarial_map'], size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        obj_maps = F.interpolate(outputs['object_heatmaps'], size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        
        res['adversarial_heatmap'] = adv_map[0, 0].cpu().numpy()
        res['category_heatmaps'] = {CLASS_NAMES[i]: obj_maps[0, i].cpu().numpy() for i in range(len(CLASS_NAMES))}

    return res

def audit_image(
    model_path,
    image_path,
    prompt="",
    vocab_path="checkpoints/vocab.json",
    return_heatmaps=False,
    timestep_value=0.0,
):
    """Convenience wrapper for auditing an image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer(vocab_path=vocab_path)
    model = CompleteMultiTaskAuditor(num_classes=5, vocab_size=len(tokenizer.word_to_idx))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return predict_single(
        model,
        tokenizer,
        image_path,
        prompt,
        return_heatmaps=return_heatmaps,
        timestep_value=timestep_value,
    )

def main():
    parser = argparse.ArgumentParser(description="Adversarial Image Auditor Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pth weights")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.json", help="Path to vocab.json")
    parser.add_argument("--image", type=str, required=True, help="Path to image to audit")
    parser.add_argument("--prompt", type=str, default="", help="Prompt used for generation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on {device}")

    # Load Tokenizer
    tokenizer = SimpleTokenizer(vocab_path=args.vocab)
    
    # Load Model
    model = CompleteMultiTaskAuditor(num_classes=5, vocab_size=len(tokenizer.word_to_idx))
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"[*] Analyzing image: {args.image}")
    results = predict_single(model, tokenizer, args.image, args.prompt, return_heatmaps=True)
    
    print("\n" + "="*40)
    print("AUDIT RESULTS")
    print("="*40)
    print(f"Adversarial:     {results['is_adversarial']} ({results['adversarial_probability']:.1%})")
    print(f"Primary Class:   {results['primary_category']}")
    print(f"Seam Quality:    {results['seam_quality_assessment']:.3f}")
    print(f"Relative Adv:    {results['relative_adversary_score']:.3f}")
    print(f"Faithfulness:    {results['text_faithfulness_score']:.3f}")
    print("-" * 40)
    print("Category Breakdown:")
    for cat, prob in results['category_probabilities'].items():
        print(f"  {cat:20s}: {prob:.1%}")
    
    # Save Heatmaps
    import cv2
    output_base = os.path.splitext(os.path.basename(args.image))[0]
    orig_img = cv2.imread(args.image)
    
    # Save Adversarial Heatmap
    if 'adversarial_heatmap' in results:
        h_map = (results['adversarial_heatmap'] * 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(orig_img, 0.6, heatmap_img, 0.4, 0)
        out_name = f"{output_base}_adv_heatmap.jpg"
        cv2.imwrite(out_name, blended)
        print(f"[*] Saved adversarial heatmap to {out_name}")

    # Save Primary Class Heatmap
    primary = results['primary_category']
    if 'category_heatmaps' in results and primary in results['category_heatmaps']:
        h_map = (results['category_heatmaps'][primary] * 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(orig_img, 0.6, heatmap_img, 0.4, 0)
        out_name = f"{output_base}_{primary.lower()}_heatmap.jpg"
        cv2.imwrite(out_name, blended)
        print(f"[*] Saved category heatmap to {out_name}")

    print("="*40)

if __name__ == "__main__":
    import numpy as np
    main()
