from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "aprs.dataset requires opencv-python (`pip install opencv-python`)."
    ) from exc



# Benchmark sample.                                                            #

@dataclass
class APRSSample:

    id: str
    split: str
    filename: str

    instruction: str
    category: str                       # EGO / UNIQ / ALLO / MULTIHOP
    instruction_zh: str = ""

    img_w: int = 0
    img_h: int = 0

    init_theta: float = 0.0             # agent's starting longitude (degrees)
    init_phi: float = 0.0               # agent's starting latitude  (degrees)

    target_theta: float = 0.0           # target box-centre longitude (degrees)
    target_phi: float = 0.0             # target box-centre latitude  (degrees)
    box_norm: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # (x,y,w,h)

    image_path: Optional[str] = None    # file path, if the panorama is on disk
    _image: Optional[np.ndarray] = field(default=None, repr=False)  # preloaded BGR

    @property
    def init_orientation(self) -> Tuple[float, float]:
        return self.init_theta, self.init_phi

    @property
    def target_orientation(self) -> Tuple[float, float]:
        return self.target_theta, self.target_phi

    def load_image(self) -> np.ndarray:
        if self._image is not None:
            return self._image
        if self.image_path and os.path.exists(self.image_path):
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read panorama: {self.image_path}")
            return img
        raise FileNotFoundError(
            f"No image available for sample {self.id!r} "
            f"(image_path={self.image_path!r}).")

    def box_pixels(self) -> Tuple[int, int, int, int]:
        x, y, w, h = self.box_norm
        return (int(x * self.img_w), int(y * self.img_h),
                int(w * self.img_w), int(h * self.img_h))


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    try:
        return float(val)
    except (TypeError, ValueError):
        return default



# Benchmark dataset.                                                           #

class APRSDataset:

    def __init__(self, root: Optional[str] = None, split: str = "train"):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.split = split
        self.root = root
        self._hf = None                          # HF dataset backend, if any
        self.samples: List[APRSSample] = []
        if root is not None:
            self._load_local(root, split)


    def _load_local(self, root: str, split: str) -> None:
        parquet = os.path.join(root, "data", "benchmark", f"{split}.parquet")
        jsonl = os.path.join(root, f"{split}.jsonl")
        meta_jsonl = os.path.join(root, "metadata", f"{split}.jsonl")
        csv_path = os.path.join(root, f"{split}.csv")
        image_dir = os.path.join(root, split)
        if os.path.exists(parquet):
            from datasets import load_dataset
            hf = load_dataset("parquet", data_files={split: parquet}, split=split)
            self._hf = hf
            self.samples = [self._sample_from_hf(i, r) for i, r in enumerate(hf)]
        elif os.path.exists(jsonl):
            self.samples = list(self._iter_jsonl(jsonl, split, image_dir))
        elif os.path.exists(meta_jsonl):
            self.samples = list(self._iter_jsonl(meta_jsonl, split, image_dir))
        elif os.path.exists(csv_path):
            self.samples = list(self._iter_csv(csv_path, split, image_dir))
        else:
            raise FileNotFoundError(
                f"No parquet/jsonl/csv annotations for split {split!r} under {root!r}.")

    @staticmethod
    def _iter_jsonl(path: str, split: str, image_dir: str):
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                fname = r.get("image") or r.get("filename") or r.get("Filename")
                fname = os.path.basename(fname) if fname else f"{split}_{i}"
                yield APRSSample(
                    id=r.get("id", f"{split}_{i}"),
                    split=split, filename=fname,
                    instruction=(r.get("instruction") or "").strip(),
                    instruction_zh=(r.get("instruction_zh") or "").strip(),
                    category=(r.get("category") or "").strip(),
                    img_w=int(r.get("width", 0)), img_h=int(r.get("height", 0)),
                    init_theta=float(r.get("init_theta", 0.0)),
                    init_phi=float(r.get("init_phi", 0.0)),
                    target_theta=float(r.get("target_theta", 0.0)),
                    target_phi=float(r.get("target_phi", 0.0)),
                    box_norm=(float(r.get("box_x", 0.0)), float(r.get("box_y", 0.0)),
                              float(r.get("box_w", 0.0)), float(r.get("box_h", 0.0))),
                    image_path=os.path.join(image_dir, fname),
                )

    @staticmethod
    def _iter_csv(path: str, split: str, image_dir: str):
        with open(path, newline="", encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                fname = row["Filename"]
                yield APRSSample(
                    id=f"{split}_{i}", split=split, filename=fname,
                    instruction=(row.get("Description") or "").strip(),
                    category=(row.get("Category") or "").strip(),
                    img_w=int(_f(row, "Img_W")), img_h=int(_f(row, "Img_H")),
                    init_theta=_f(row, "Pt_Theta"), init_phi=_f(row, "Pt_Phi"),
                    target_theta=_f(row, "Box_Theta"), target_phi=_f(row, "Box_Phi"),
                    box_norm=(_f(row, "Box_X_Norm"), _f(row, "Box_Y_Norm"),
                              _f(row, "Box_W_Norm"), _f(row, "Box_H_Norm")),
                    image_path=os.path.join(image_dir, fname),
                )


    @classmethod
    def from_parquet(cls, files: "str | Sequence[str]", split: str = "train"
                     ) -> "APRSDataset":
        from datasets import load_dataset
        data_files = [files] if isinstance(files, str) else list(files)
        hf = load_dataset("parquet", data_files={split: data_files}, split=split)
        return cls._from_hf(hf, split)

    @classmethod
    def from_hub(cls, repo_id: str = "FudanCVL/APRS_dataset",
                 split: str = "train") -> "APRSDataset":
        from datasets import load_dataset
        # Load from HuggingFace Hub - the parquet files are in data/ directory
        hf = load_dataset(
            repo_id,
            data_files={split: f"data/{split}.parquet"},
            split=split
        )
        return cls._from_hf(hf, split)

    @classmethod
    def _from_hf(cls, hf, split: str) -> "APRSDataset":
        obj = cls(root=None, split=split)
        obj._hf = hf
        obj.samples = [obj._sample_from_hf(i, r) for i, r in enumerate(hf)]
        return obj

    def _sample_from_hf(self, i: int, r: Dict[str, Any]) -> APRSSample:
        img = None
        pil = r.get("image")
        if pil is not None and not isinstance(pil, (str, bytes)):
            # PIL.Image -> BGR ndarray (lazy: only when iterating rows)
            img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Support both new CSV column names and standard field names
        filename = r.get("Filename") or r.get("filename") or r.get("id", f"{self.split}_{i}")
        instruction = r.get("Description") or r.get("instruction") or ""
        instruction_zh = r.get("instruction_zh") or ""
        category = r.get("Category") or r.get("category") or ""

        img_w = int(r.get("Img_W") or r.get("width") or 0)
        img_h = int(r.get("Img_H") or r.get("height") or 0)

        init_theta = float(r.get("Pt_Theta") or r.get("init_theta") or 0.0)
        init_phi = float(r.get("Pt_Phi") or r.get("init_phi") or 0.0)
        target_theta = float(r.get("Box_Theta") or r.get("target_theta") or 0.0)
        target_phi = float(r.get("Box_Phi") or r.get("target_phi") or 0.0)

        box_x = float(r.get("Box_X_Norm") or r.get("box_x") or 0.0)
        box_y = float(r.get("Box_Y_Norm") or r.get("box_y") or 0.0)
        box_w = float(r.get("Box_W_Norm") or r.get("box_w") or 0.0)
        box_h = float(r.get("Box_H_Norm") or r.get("box_h") or 0.0)

        return APRSSample(
            id=r.get("id", f"{self.split}_{i}"),
            split=self.split,
            filename=filename,
            instruction=instruction,
            instruction_zh=instruction_zh,
            category=category,
            img_w=img_w,
            img_h=img_h,
            init_theta=init_theta,
            init_phi=init_phi,
            target_theta=target_theta,
            target_phi=target_phi,
            box_norm=(box_x, box_y, box_w, box_h),
            _image=img,
        )


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> APRSSample:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def by_category(self, category: str) -> List[APRSSample]:
        c = category.upper()
        return [s for s in self.samples if s.category.upper() == c]

    def category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.samples:
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts
