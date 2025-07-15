import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import imageio.v2 as imageio  # PNG/JPEG fallback; replace by nibabel for NIfTI

"""dataset_cpa.py
CPADataset — Real multimodal dataset loader for CPAT
===================================================
Expected directory layout
------------------------
```
DATASET_ROOT/
├── meta.csv                # tabular file with one row per patient
├── ct/                     # CT slices or volumes
│   ├── <id>.png            # e.g. 224×224 pre‑resampled PNG
│   └── ...
├── ehr_text/               # plain‑text files or leave empty if text in meta.csv
│   ├── <id>.txt
│   └── ...
```
`meta.csv` **must** contain the following columns:
* **id**      Unique patient identifier (matches CT filename).
* **label**   Binary 0/1 — survival(0) or death(1).
* **prior\_\*** Columns forming the prior vector (e.g. `prior_age`, `prior_apache`).
* **lab\_\***  Laboratory test columns.
* (optional) **text** If clinical notes embedded directly in CSV.

Modify column patterns in `lab_cols` / `prior_cols` to match your schema.
"""

class CPADataset(Dataset):
    def __init__(
        self,
        root: str,
        csv_file: str = "meta.csv",
        ct_subdir: str = "ct",
        text_subdir: str | None = None,
        lab_cols_pattern: str = "lab_",
        prior_cols_pattern: str = "prior_",
        transform_ct=None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        self.root = root
        self.df = pd.read_csv(os.path.join(root, csv_file))

        # Identify column groups
        self.lab_cols   = [c for c in self.df.columns if c.startswith(lab_cols_pattern)]
        self.prior_cols = [c for c in self.df.columns if c.startswith(prior_cols_pattern)]

        if text_subdir is None and "text" not in self.df.columns:
            raise ValueError("Provide text_subdir or a 'text' column in CSV")
        self.text_subdir = text_subdir
        self.ct_dir = os.path.join(root, ct_subdir)
        self.transform_ct = transform_ct
        self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def _load_ct(self, pid: str):
        """Load 2‑D CT slice as [1,H,W] tensor. Replace with 3‑D volume if needed."""
        path_png = os.path.join(self.ct_dir, f"{pid}.png")
        img = imageio.imread(path_png).astype(self.dtype) / 255.0  # [H,W]
        if self.transform_ct:
            img = self.transform_ct(img)
        img = torch.from_numpy(img).unsqueeze(0)  # [1,H,W]
        return img

    def _load_text(self, pid: str):
        if self.text_subdir is not None:
            path_txt = os.path.join(self.root, self.text_subdir, f"{pid}.txt")
            with open(path_txt, "r", encoding="utf-8") as f:
                return f.read()
        return self.df.loc[self.df.id == pid, "text"].values[0]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pid = str(row.id)

        ct    = self._load_ct(pid)
        text  = self._load_text(pid)
        lab   = torch.tensor(row[self.lab_cols].values.astype(self.dtype))
        prior = torch.tensor(row[self.prior_cols].values.astype(self.dtype))
        label = torch.tensor(row.label, dtype=torch.float32)

        return ct, text, lab, prior, label

# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":
    ds = CPADataset(root="/path/to/dataset", text_subdir="ehr_text")
    print("Num samples:", len(ds))
    sample = ds[0]
    for k, v in zip(["ct", "text", "lab", "prior", "label"], sample):
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v[:60] + "...")

