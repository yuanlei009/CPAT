import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -------- Project modules ---------
from swin_encoder import SwinCTEncoder
from clinical_bert_encoder import ClinicalBERTEncoder
from lab_transformer_encoder import LabTransformerEncoder
from ot_alignment import OTAligner
from tucker_decomposition import TuckerDecomposer
from prior_guided_attention import PriorGuidedAttention
from modality_fusion import ModalityFusion
from dataset_cpa import CPADataset  # <-- REAL dataset loader

# ===============================================
#  CPATBinary model (unchanged)
# ===============================================
class CPATBinary(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        D = cfg['model']['embed_dim']
        prior_dim = cfg['model']['prior_dim']

        # Encoders
        self.ct_enc  = SwinCTEncoder(pretrained=False, model_name=cfg['ct']['backbone'])
        self.txt_enc = ClinicalBERTEncoder(output_cls=True, freeze=True)
        self.lab_enc = LabTransformerEncoder(num_features=cfg['lab']['num_tests'], d_model=D)

        # Alignment / Reconstruction / Attention / Fusion
        self.ot     = OTAligner(in_dim=D, proj_dim=D)
        self.tucker = TuckerDecomposer(feat_dim=D)
        self.attn   = PriorGuidedAttention(prior_dim=prior_dim, embed_dim=D)
        self.fusion = ModalityFusion(embed_dim=D, prior_dim=prior_dim, dynamic=True)

        # Classification head
        self.head      = nn.Linear(D, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, ct, text, lab, prior, label=None):
        # Encode each modality
        z_ct  = self.ct_enc(ct)            # [B,C,h,w]
        z_ct  = z_ct.mean(dim=[2, 3])      # -> [B,D]
        z_txt = self.txt_enc(text)         # [B,D]
        z_lab = self.lab_enc(lab)          # [B,D]

        # OT alignment
        z_align, ot_loss = self.ot(z_ct, z_txt, z_lab)

        # Tucker reconstruction
        z_rec, rec_loss = self.tucker(z_ct, z_txt, z_lab)

        # Prior‑guided attention
        z_att, _ = self.attn(prior, z_align, z_rec, z_lab)

        # Fusion
        z_fused, _ = self.fusion(z_att, z_rec, z_align, prior)

        # Classification
        logits = self.head(z_fused).squeeze(-1)
        loss = None
        if label is not None:
            loss_cls = self.criterion(logits, label.squeeze())
            loss = loss_cls + 0.1 * ot_loss + 0.1 * rec_loss
        return logits, loss

# ===============================================
#  Training loop util
# ===============================================

def train_one_epoch(model, loader, optim, device):
    model.train(); total = 0
    for ct, text, lab, prior, label in loader:
        ct, lab, prior, label = ct.to(device), lab.to(device), prior.to(device), label.to(device)
        logits, loss = model(ct, text, lab, prior, label)
        optim.zero_grad(); loss.backward(); optim.step()
        total += loss.item() * ct.size(0)
    return total / len(loader.dataset)

# ===============================================
#  Main entry
# ===============================================
if __name__ == "__main__":
    cfg = {
        'model': {'embed_dim': 128, 'prior_dim': 64},
        'ct':    {'backbone': 'swin_tiny_patch4_window7_224'},
        'lab':   {'num_tests': 12},
        'train': {'batch': 4, 'epochs': 5, 'lr': 1e-4},
        'data':  {'root': '/data/CPA', 'text_subdir': 'ehr_text'}  # <-- modify path!
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build real dataset
    train_ds = CPADataset(root=cfg['data']['root'], text_subdir=cfg['data']['text_subdir'])
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch'], shuffle=True, num_workers=4)

    model = CPATBinary(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])

    for epoch in range(cfg['train']['epochs']):
        loss = train_one_epoch(model, train_loader, optim, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}")
