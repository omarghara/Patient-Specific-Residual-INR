"""SLAM dataset: minimal downloader + per-slice loader.

Mirrors the download / preparation logic of ``laps.slam`` (Stanford Digital
Repository, PURL rq296rb2765) so we can pull data without installing the full
LAPS environment. The dataset downloads over plain HTTP -- no credentials
needed (HuggingFace login in LAPS is only for their diffusion models).

Layout produced (under ``DATA_DIR``):
    slam/                 raw volumes + train.csv / test.csv
    slam-test/            per-slice: <scan>_<sss>/{recon.npy, prior.npy, data.pt}
    slam-test.csv         one row per slice (incl. change_extent, is_middle_slice)

Each test slice provides everything the residual-INR reconstruction needs:
complex k-space and coil maps ``(Nc, Kx, Ky)``, sampling mask ``(Kx, Ky)``,
plus the center-padded ``(Nx, Ny)`` reference and registered magnitude prior.
"""

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

SLAM_SDR_URL = "https://stacks.stanford.edu/file/rq296rb2765"

# Default to <repo>/data (three parents up: data/ -> presinr/ -> src/ -> repo/).
DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _dirs(data_dir: Path):
    return {
        "slam": data_dir / "slam",
        "test": data_dir / "slam-test",
        "train": data_dir / "slam-train",
    }


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def pull_metadata(data_dir: Path = DATA_DIR, minimal: bool = True, override: bool = False):
    slam = _dirs(data_dir)["slam"]
    slam.mkdir(parents=True, exist_ok=True)
    for f in ["README.md", "example.py", "test.csv", "train.csv"]:
        dst = slam / f
        if override or not dst.exists():
            print(f"  metadata: {f}")
            urllib.request.urlretrieve(f"{SLAM_SDR_URL}/{f}", dst)
    if minimal:
        test_csv, train_csv = slam / "test.csv", slam / "train.csv"
        dt = pd.read_csv(test_csv)
        # Row 6 is the minimal test example used by LAPS (has k-space).
        idx = 6 if len(dt) > 6 else 0
        dt = dt.iloc[[idx]].reset_index(drop=True)
        dt.loc[:, "index"] = range(len(dt))
        dt.to_csv(test_csv, index=False)
        dr = pd.read_csv(train_csv).iloc[0:5].reset_index(drop=True)
        dr.loc[:, "index"] = range(len(dr))
        dr.to_csv(train_csv, index=False)
        print(f"  minimal: {len(dt)} test scan, {len(dr)} train scans")


def pull_volumes(split: str, data_dir: Path = DATA_DIR, load_ksp: bool = False, override: bool = False):
    slam = _dirs(data_dir)["slam"]
    df = pd.read_csv(slam / f"{split}.csv")
    keys = ["recon_path", "prior_path"] + (["ksp_path"] if load_ksp else [])
    base_url = f"{SLAM_SDR_URL}/recon/{split}"
    split_dir = slam / "recon" / split
    split_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for key in keys:
        if key in df.columns:
            files += [p for p in df[key].dropna().unique()]
    print(f"  {split}: downloading {len(files)} volume file(s)")
    for path in tqdm(files, desc=f"SLAM {split}"):
        dst = split_dir / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if override or not dst.exists():
            urllib.request.urlretrieve(f"{base_url}/{path}", dst)


def download_minimal(data_dir: Path = DATA_DIR):
    """Download the minimal SLAM set: 5 train volumes + 1 test scan with k-space."""
    print("Downloading SLAM (minimal)...")
    pull_metadata(data_dir, minimal=True, override=True)
    pull_volumes("train", data_dir, load_ksp=False, override=True)
    pull_volumes("test", data_dir, load_ksp=True, override=True)
    prepare_test(data_dir)
    print("SLAM minimal download complete.")


# --------------------------------------------------------------------------- #
# Prepare per-slice test data (mirrors laps.slam.prepare_slam_test)
# --------------------------------------------------------------------------- #
def prepare_test(data_dir: Path = DATA_DIR):
    slam = _dirs(data_dir)["slam"]
    root = _dirs(data_dir)["test"]
    root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(slam / "test.csv")

    keep_cols = [
        "subj_index", "change_extent", "scan_plane", "scan_type", "quality",
        "Nc", "Kx", "Ky", "Nx", "Ny", "Nz", "Rro", "Rpe", "AccelNumDim",
    ]
    rows = []
    for scan_idx, row in tqdm(df.iterrows(), total=len(df), desc="prepare test"):
        vol = row["recon_path"].replace("/recon.npz", "")
        vdir = slam / "recon" / "test" / vol
        recon = np.load(vdir / "recon.npz")["arr_0"]     # (H, W, S) complex
        prior = np.load(vdir / "prior.npz")["arr_0"]     # (H, W, S)
        data = np.load(vdir / "data.npz")
        ksp = torch.from_numpy(data["ksp"])              # (Nc, Kx, Ky, S)
        mps = torch.from_numpy(data["mps"])              # (Nc, Kx, Ky, S)
        mask = torch.from_numpy(data["mask"])            # (Kx, Ky, S)

        start, end = int(row["slc_start_idx"]), int(row["slc_end_idx"])
        recon, prior = recon[:, :, start:end], prior[:, :, start:end]
        ksp, mps, mask = ksp[..., start:end], mps[..., start:end], mask[..., start:end]
        mid = recon.shape[2] // 2

        for s in range(recon.shape[2]):
            folder = Path(str(root / vol) + f"_{s:06d}")
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "recon.npy", recon[..., s])
            np.save(folder / "prior.npy", prior[..., s])
            torch.save(
                {"ksp": ksp[..., s].clone(), "mps": mps[..., s].clone(), "mask": mask[..., s].clone()},
                folder / "data.pt",
            )
            rec = {
                "index": len(rows),
                "scan_index": scan_idx,
                "slice_index": s,
                "recon_path": str((folder / "recon.npy").relative_to(root)),
                "prior_path": str((folder / "prior.npy").relative_to(root)),
                "data_path": str((folder / "data.pt").relative_to(root)),
                "is_middle_slice": (s == mid),
            }
            for c in keep_cols:
                rec[c] = row[c] if c in df.columns else None
            rows.append(rec)

    out_csv = str(root) + ".csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  wrote {len(rows)} test slices -> {out_csv}")


# --------------------------------------------------------------------------- #
# Image-only path (magnitude recon + prior, no k-space) for image-space work
# --------------------------------------------------------------------------- #
def fetch_test_image_scans(recon_paths, data_dir: Path = DATA_DIR, override: bool = False):
    """Download recon + prior (magnitude, NO k-space) for the given test scans
    and prepare per-slice image data. ``recon_paths`` are ``recon_path`` values
    from the full test.csv (e.g. ``EX001/SCN04_Ax_T2_2D/recon.npz``)."""
    slam = _dirs(data_dir)["slam"]
    slam.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{SLAM_SDR_URL}/test.csv", slam / "test.csv")  # full manifest
    df = pd.read_csv(slam / "test.csv")
    sel = df[df["recon_path"].isin(recon_paths)].reset_index(drop=True)
    assert len(sel) == len(recon_paths), f"found {len(sel)}/{len(recon_paths)} requested scans"

    base_url = f"{SLAM_SDR_URL}/recon/test"
    split_dir = slam / "recon" / "test"
    for _, row in sel.iterrows():
        for key in ["recon_path", "prior_path"]:
            path = row[key]
            if pd.isna(path):
                continue
            dst = split_dir / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if override or not dst.exists():
                print(f"  downloading {path}")
                urllib.request.urlretrieve(f"{base_url}/{path}", dst)
    prepare_test_images(sel, data_dir)
    return sel


def prepare_test_images(df, data_dir: Path = DATA_DIR):
    """Materialize per-slice magnitude recon + prior into ``slam-test-img/``.
    Accumulates across calls (existing scans are replaced, not duplicated)."""
    root = data_dir / "slam-test-img"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in df.iterrows():
        vol = row["recon_path"].replace("/recon.npz", "")
        vdir = data_dir / "slam" / "recon" / "test" / vol
        recon = np.load(vdir / "recon.npz")["arr_0"]     # (H, W, S)
        prior = np.load(vdir / "prior.npz")["arr_0"]
        start, end = int(row["slc_start_idx"]), int(row["slc_end_idx"])
        recon, prior = recon[:, :, start:end], prior[:, :, start:end]
        mid = recon.shape[2] // 2
        for s in range(recon.shape[2]):
            folder = Path(str(root / vol) + f"_{s:06d}")
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "recon.npy", recon[..., s])
            np.save(folder / "prior.npy", prior[..., s])
            rows.append({
                "scan_index": int(row["index"]),
                "slice_index": s,
                "change_extent": int(row["change_extent"]),
                "is_middle_slice": (s == mid),
                "scan_type": row["scan_type"],
                "recon_path": str((folder / "recon.npy").relative_to(root)),
                "prior_path": str((folder / "prior.npy").relative_to(root)),
            })
    new = pd.DataFrame(rows)
    out = str(root) + ".csv"
    if Path(out).exists():
        old = pd.read_csv(out)
        old = old[~old["scan_index"].isin(new["scan_index"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.insert(0, "index", range(len(new)))
    new.to_csv(out, index=False)
    print(f"  prepared {len(rows)} slices; {len(new)} total -> {out}")


class SlamTestImageSlices(Dataset):
    """Per-slice magnitude recon + prior (no k-space) for image-space experiments."""

    def __init__(self, data_dir: Path = DATA_DIR, change_extent=None, scan_index=None, normalize=True):
        self.root = _dirs(Path(data_dir))["slam"].parent / "slam-test-img"
        self.df = pd.read_csv(str(self.root) + ".csv")
        if change_extent is not None:
            self.df = self.df[self.df["change_extent"] == change_extent].reset_index(drop=True)
        if scan_index is not None:
            self.df = self.df[self.df["scan_index"] == scan_index].reset_index(drop=True)
        self.normalize = normalize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        recon = np.abs(np.asarray(np.load(self.root / row["recon_path"]))).astype(np.float32)
        prior = np.abs(np.asarray(np.load(self.root / row["prior_path"]))).astype(np.float32)
        recon_t = torch.from_numpy(recon)
        prior_t = torch.from_numpy(prior)
        if self.normalize:
            recon_t = recon_t / (float(torch.quantile(recon_t.reshape(-1), 0.999)) + 1e-8)
            prior_t = prior_t / (float(torch.quantile(prior_t.reshape(-1), 0.999)) + 1e-8)
        return {
            "recon": recon_t, "prior": prior_t,
            "change_extent": int(row["change_extent"]),
            "scan_index": int(row["scan_index"]),
            "slice_index": int(row["slice_index"]),
            "is_middle_slice": bool(row["is_middle_slice"]),
        }


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SlamTestSlices(Dataset):
    """Per-slice SLAM test set for reconstruction.

    Returns tensors ``ksp/mps (Nc,Kx,Ky)``, ``mask (Kx,Ky)``, center-padded
    ``recon (Nx,Ny)`` and ``prior (Nx,Ny)``, as well as exact center-cropped
    ``recon_native/prior_native (Kx,Ky)`` views. ``ksp_scale`` is the robust
    intensity scale used to normalize k-space to an image range near ``[0,1]``.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        middle_only: bool = False,
        normalize: bool = True,
        change_extent: Optional[int] = None,
    ):
        self.root = _dirs(Path(data_dir))["test"]
        self.df = pd.read_csv(str(self.root) + ".csv")
        if middle_only and "is_middle_slice" in self.df.columns:
            self.df = self.df[self.df["is_middle_slice"]].reset_index(drop=True)
        if change_extent is not None and "change_extent" in self.df.columns:
            self.df = self.df[self.df["change_extent"] == change_extent].reset_index(drop=True)
        self.normalize = normalize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        recon = np.load(self.root / row["recon_path"])
        prior = np.load(self.root / row["prior_path"])
        data = torch.load(self.root / row["data_path"], weights_only=True)
        ksp = data["ksp"].to(torch.complex64)
        mps = data["mps"].to(torch.complex64)
        mask = data["mask"].to(torch.float32)

        recon_t = torch.from_numpy(np.asarray(recon)).to(torch.complex64)
        prior_t = torch.from_numpy(np.abs(np.asarray(prior))).to(torch.float32)

        native_shape = tuple(ksp.shape[-2:])
        stored_shape = tuple(recon_t.shape[-2:])
        if tuple(mps.shape[-2:]) != native_shape or tuple(mask.shape) != native_shape:
            raise ValueError(
                "SLAM native-grid mismatch: "
                f"ksp={tuple(ksp.shape)}, mps={tuple(mps.shape)}, mask={tuple(mask.shape)}"
            )
        if tuple(prior_t.shape) != stored_shape:
            raise ValueError(
                f"SLAM stored-grid mismatch: recon={stored_shape}, prior={tuple(prior_t.shape)}"
            )
        if any(native > stored for native, stored in zip(native_shape, stored_shape)):
            raise ValueError(
                f"native grid {native_shape} cannot exceed stored grid {stored_shape}"
            )

        # Raw k-space, reference recon, and prior arrive at three unrelated
        # absolute scales. Normalize k-space by the *zero-filled adjoint* scale so
        # the reconstruction target is ~[0, 1] (matches the INR output range), and
        # normalize recon / prior each by their own robust scale for evaluation.
        from ..fft import ifftc

        zf = torch.sum(ifftc(ksp * mask, dim=(-2, -1)) * torch.conj(mps), dim=0)
        ksp_scale = float(torch.quantile(zf.abs().reshape(-1), 0.999)) + 1e-8
        if self.normalize:
            ksp = ksp / ksp_scale
            recon_t = recon_t / (float(torch.quantile(recon_t.abs().reshape(-1), 0.999)) + 1e-8)
            prior_t = prior_t / (float(torch.quantile(prior_t.reshape(-1), 0.999)) + 1e-8)

        from ..utils import center_crop_to

        recon_native = center_crop_to(recon_t, native_shape)
        prior_native = center_crop_to(prior_t, native_shape)

        return {
            "ksp": ksp, "mps": mps, "mask": mask,
            "recon": recon_t, "prior": prior_t,
            "recon_native": recon_native, "prior_native": prior_native,
            "native_shape": native_shape, "stored_shape": stored_shape,
            "change_extent": int(row["change_extent"]) if "change_extent" in row and not pd.isna(row["change_extent"]) else -1,
            "ksp_scale": ksp_scale,
            "index": int(row["index"]),
        }
