"""ortho_unified_core.py — Paper 3: GeoFiLM (M4) + Ortho-Unified Closed-Loop Bridge.

Berdasarkan post-mortem empiris dari `B1_M4_CanonBridge` (notebook 05, yang sudah mengungguli
Gautam_Qwen2.5VL_7B: 85,96% vs 85,50% Qwen-Judge, p=0,011; 87,50% vs 87,02% Llama-Judge, p=0,0053)
dan `E_CLBRIDGE_r32` (82,18% pada polyp_size), modul ini mempertahankan 100% resep M4 + CL-Bridge
(split, prompt, target prosa murni tanpa slot, loss VQA CE bawaan tanpa focal, LoRA r=32, HP,
pemulihan RNG sehingga step 0 == M4 persis dengan val_L_VQA = 5,0737), lalu memperbaiki 5 leher
botol matematis yang ditemukan pada diagnostik `[E4]` notebook 05:

  1. [Fix 1 — 9D Spatio-Geometric Moments + 2D Fourier Coordinates per Category]:
     Pada `B1_M4_CanonBridge`, 16 kepala kanonik hanya menerima 4 statistik area
     `[gate_mean, gate_std, s_max, ent]` TANPA koordinat 2D (mu_x, mu_y, cov). Akibatnya,
     10 kategori non-lokasi mencapai akurasi kepala 90,8%–99,8%, sedangkan 3 kategori lokasi
     tertinggal di 55,2%–69,3%.
     -> `OrthoUnifiedBridge` menghitung momen geometrik 9D penuh per kategori:
        [mu_x, mu_y, cov_xx, cov_yy, cov_xy, gate_mean, gate_std, s_max, ent]
        ditambah fitur koordinat 2D Fourier terpelajar dari (mu_x, mu_y) ke setiap `fc[ci]`
        dan `embed[ci]`.

  2. [Fix 2 — Token-Level Query Cross-Attention & Supervised Query-Aspect Router r_c(Q)]:
     `B1_M4_CanonBridge` menjumlahkan ke-16 vektor kategori secara buta ke `tok0` (derau 15:1
     saat pertanyaan hanya menanyakan 1–2 aspek), sehingga gerbang tertahan di ~0,07.
     -> `OrthoUnifiedBridge` memprediksi gerbang relevansi pertanyaan per kategori r_c(Q) in (0,1)
        dari `query_embeds` (disupervisi dengan BCE terhadap `canon_m`), serta melakukan
        Query-to-Category Cross-Attention agar kategori yang ditanyakan diperkuat 2x dan
        kategori yang tidak ditanyakan diredam.

  3. [Fix 3 — Exact FP32 Gram-Schmidt Orthogonal Subspace Projection]:
     Menambahkan vektor bridge secara langsung di atas `e_total` dan `pe_2d` milik GeoFiLM
     memicu tabrakan fitur (gerbang spasial negatif -0,11 / -0,13 pada B1).
     -> `OrthoUnifiedBridge` memproyeksikan injeksi global `glob` ke komplemen ortogonal dari
        `e_total` dan memproyeksikan injeksi spasial `inj_p` ke komplemen ortogonal dari `pe_2d_p`
        di dalam pulau FP32:
           <glob_perp, e_total> = 0,   <inj_p_perp, pe_2d_p> = 0  (eksak di setiap step).

  4. [Fix 4 — Dedicated Monotonic CORN + Squared EMD Ordinal Head untuk `polyp_size`]:
     BCE biasa mengabaikan urutan fisik ketat `<5mm < 5-10mm < 11-20mm < >20mm`.
     -> `OrthoUnifiedBridge` memasang kepala Conditional Ordinal Regression Network (CORN)
        monotonik + Squared Earth Mover's Distance (EMD) khusus untuk `polyp_size` dan
        menyuntikkan probabilitas kumulatif ordinal langsung ke jembatan metrik `polyp_size`.

  5. [Fix 5 — Per-Patch Coordinate-Aware Residual FiLM (Zero-Init Identity pada Step 0)]:
     Modulasi FiLM residual per-patch berbasis koordinat 2D + konteks ortogonal dengan proyeksi
     keluaran berinisialisasi nol sehingga pada step 0 keluaran == M4 bit-for-bit.
"""
import ast
import json
import math
import re
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CORE_VERSION = "ortho-unified-2026-11-15"

# Urutan fisik ketat untuk polyp_size (dipakai oleh kepala CORN + EMD)
POLYP_SIZE_ORDERED_TIERS = ["<5mm", "5-10mm", "11-20mm", ">20mm"]

# Karakter yang TIDAK BOLEH ada di atom kanonik.
_FORBIDDEN_IN_ATOM = set("{}\"':?[]")


# =============================================================================
# 1. UTILITAS DATA (100% kompatibel dengan geofilm_bridge_core.py)
# =============================================================================
def as_list(x):
    """`original` -> list jawaban (nilai kunci "a")."""
    if x is None:
        return []
    if isinstance(x, dict):
        return [str(x.get("a", ""))] if "a" in x else []
    if isinstance(x, (list, tuple, np.ndarray)):
        out = []
        for v in x:
            if isinstance(v, (dict, list, tuple, np.ndarray)):
                out.extend(as_list(v))
            else:
                out.append(str(v))
        return out
    s = str(x).strip()
    if not s:
        return []
    for parse in (json.loads, ast.literal_eval):
        try:
            v = parse(s)
        except Exception:
            continue
        if isinstance(v, (dict, list, tuple)):
            return as_list(v)
        return [str(v)]
    return [s]


def atoms(s):
    """Pecah satu jawaban menjadi atom (mis. 'center; upper-center' -> 2 atom)."""
    parts = re.split(r"[,;]| and ", str(s).lower())
    return [p.strip(" .") for p in parts if p.strip(" .")]


def as_class_list(x):
    """`question_class` (list/ndarray/str) -> list nama kelas."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)
    return [s]


def aspect_pairs(qclass, original):
    """Pasangkan question_class[i] dengan jawaban original[i]. None bila tak dapat dipastikan."""
    qs = as_class_list(qclass)
    og = as_list(original)
    if not qs or not og:
        return None
    if len(qs) == len(og):
        return list(zip(qs, og))
    if len(qs) == 1:
        return [(qs[0], o) for o in og]
    return None


def selfcheck_as_list():
    """Gerbang: as_list WAJIB mengembalikan jawaban bersih pada format asli dataset."""
    cases = {
        '[\n{\n"q": "What is the size of the polyp?",\n"a": "11-20mm"\n}\n]': ["11-20mm"],
        '[{"q": "Where?", "a": "center; upper-center"}, {"q": "Text?", "a": "yes"}]':
            ["center; upper-center", "yes"],
        "{'q': 'Is there text?', 'a': 'no'}": ["no"],
        "plain answer": ["plain answer"],
    }
    for inp, exp in cases.items():
        got = as_list(inp)
        assert got == exp, f"as_list({inp!r}) = {got!r}, harus {exp!r}"
    return True


# =============================================================================
# 2. SPESIFIKASI KANONIK & ORDINAL (ditemukan dari data TRAIN)
# =============================================================================
def discover_canonical_specs(df, min_count=20, min_cover=0.99, verbose=True):
    """-> OrderedDict {kategori: {"atoms": [...], "multi": bool, ...}} (urut nama, deterministik)."""
    cnt, n_asp, max_k, n_bad = {}, Counter(), Counter(), 0
    for q, o in zip(df["question_class"], df["original"]):
        P = aspect_pairs(q, o)
        if P is None:
            n_bad += 1
            continue
        for c, a in P:
            A = atoms(a)
            cnt.setdefault(c, Counter()).update(A)
            n_asp[c] += 1
            max_k[c] = max(max_k[c], len(A))

    specs, report = OrderedDict(), []
    for c in sorted(cnt):
        vocab = sorted(a for a, n in cnt[c].items() if n >= min_count)
        tot = sum(cnt[c].values())
        cover = sum(cnt[c][a] for a in vocab) / max(tot, 1)
        keep = len(vocab) >= 2 and cover >= min_cover
        why = "OK" if keep else ("jawaban tunggal" if len(vocab) < 2 else f"cover {cover:.3f}<{min_cover}")
        report.append((c, n_asp[c], len(vocab), round(100 * cover, 2), max_k[c] > 1, why))
        if keep:
            specs[c] = {"atoms": vocab, "multi": bool(max_k[c] > 1),
                        "n_aspects": int(n_asp[c]), "cover": float(cover)}
    if verbose:
        print(f"Spesifikasi kanonik (min_count={min_count}, min_cover={min_cover}) | baris tak sejajar: {n_bad}")
        print(f"  {'kategori':24s} {'aspek':>7s} {'|V|':>4s} {'cover%':>7s} {'multi':>6s}  status")
        for c, n, v, cv, mu, why in report:
            print(f"  {c:24s} {n:7d} {v:4d} {cv:7.2f} {str(mu):>6s}  {why}")
        print(f"  -> {len(specs)} kategori, {sum(len(s['atoms']) for s in specs.values())} atom total")
    assert_clean_specs(specs)
    return specs


def assert_clean_specs(specs):
    """Gerbang anti-regresi bug U4: tak boleh ada sisa repr/JSON di dalam atom."""
    dirty = [(c, a) for c, s in specs.items() for a in s["atoms"] if set(a) & _FORBIDDEN_IN_ATOM]
    assert not dirty, ("Kosakata kanonik TERCEMAR (bug as_list lama?) — hentikan sebelum training:\n"
                       + "\n".join(f"  {c}: {a!r}" for c, a in dirty[:20]))
    return True


class CanonLabeler:
    """Mengubah (question_class, original) -> target datar untuk kepala kanonik + target ordinal CORN."""

    def __init__(self, specs):
        self.specs = specs
        self.cats = list(specs.keys())
        self.multi = [bool(specs[c]["multi"]) for c in self.cats]
        self.slices, off = [], 0
        self.atom_index = {}
        for ci, c in enumerate(self.cats):
            A = specs[c]["atoms"]
            self.slices.append((off, off + len(A)))
            for j, a in enumerate(A):
                self.atom_index[(c, a)] = off + j
            off += len(A)
        self.T, self.C = off, len(self.cats)
        self.cat_index = {c: i for i, c in enumerate(self.cats)}
        self.size_cat_idx = self.cat_index.get("polyp_size", -1)
        self.size_tier2rank = {t: r for r, t in enumerate(POLYP_SIZE_ORDERED_TIERS)}
        # Peta indeks lokal atom di dalam kategori polyp_size untuk keempat kelas ukuran fisik
        self.size_rank2local = []
        if self.size_cat_idx >= 0:
            sz_atoms = specs["polyp_size"]["atoms"]
            for t in POLYP_SIZE_ORDERED_TIERS:
                self.size_rank2local.append(sz_atoms.index(t) if t in sz_atoms else -1)

    def encode(self, qclass, original):
        """-> list[(cat_idx, [flat_idx,...])] untuk satu baris."""
        P = aspect_pairs(qclass, original)
        if not P:
            return []
        out, seen = [], set()
        for c, a in P:
            if c not in self.cat_index or c in seen:
                continue
            A = atoms(a)
            idx = [self.atom_index.get((c, t)) for t in A]
            if not A or any(i is None for i in idx):
                continue
            ci = self.cat_index[c]
            if not self.multi[ci] and len(set(idx)) != 1:
                continue
            out.append((ci, sorted(set(idx))))
            seen.add(c)
        return out

    def to_tensors(self, enc_list):
        """-> (y [B, T], m [B, C]) kompatibel dengan B1_M4_CanonBridge."""
        B = len(enc_list)
        y = torch.zeros(B, self.T, dtype=torch.float32)
        m = torch.zeros(B, self.C, dtype=torch.float32)
        for b, enc in enumerate(enc_list):
            for ci, idx in enc:
                y[b, idx] = 1.0
                m[b, ci] = 1.0
        return y, m

    def extract_size_ordinal_targets(self, y: torch.Tensor, m: torch.Tensor):
        """Ekstrak target ordinal polyp_size langsung dari tensor (y, m) tanpa mengubah format dataset!
        Mengembalikan:
          - ord_y: (B,) long dalam {0, 1, 2, 3} corresponding to (<5mm, 5-10mm, 11-20mm, >20mm)
          - ord_m: (B,) float mask (1.0 bila baris menanyakan polyp_size dengan tier fisik valid)
        """
        B = y.shape[0]
        device = y.device
        ord_y = torch.zeros(B, dtype=torch.long, device=device)
        ord_m = torch.zeros(B, dtype=torch.float32, device=device)
        if self.size_cat_idx < 0 or any(loc < 0 for loc in self.size_rank2local):
            return ord_y, ord_m

        s0, _ = self.slices[self.size_cat_idx]
        cat_active = m[:, self.size_cat_idx] > 0.5
        if not cat_active.any():
            return ord_y, ord_m

        # Ambil sub-kolom keempat tier fisik sesuai urutan <5mm, 5-10mm, 11-20mm, >20mm
        tier_cols = [s0 + loc for loc in self.size_rank2local]
        tier_hits = y[:, tier_cols] > 0.5  # (B, 4)
        has_tier = cat_active & tier_hits.any(dim=1)
        if has_tier.any():
            # Konvensi klinis: bila ada >1 polip, tier ukuran terbesar menentukan risiko klinis
            ranks = torch.arange(4, device=device).unsqueeze(0)  # (1, 4)
            max_rk = (tier_hits.long() * (ranks + 1)).argmax(dim=1)
            ord_y = torch.where(has_tier, max_rk, ord_y)
            ord_m = has_tier.float()
        return ord_y, ord_m


# =============================================================================
# 3. GeoFiLMModule — SALINAN PERSIS notebook 01 [A8] & geofilm_bridge_core.py
# =============================================================================
class GeoFiLMModule(nn.Module):
    def __init__(self, d_in=768, d_fourier=32, d_rank=64,
                 use_geope=True, use_diff=True, gate_mode="film"):
        super().__init__()
        assert gate_mode in ("additive", "scalar", "film"), f"Unknown gate_mode: {gate_mode}"
        self.d_in = d_in
        self.d_fourier = d_fourier
        self.d_rank = d_rank
        self.use_geope = use_geope
        self.use_diff = use_diff
        self.gate_mode = gate_mode
        self.scale = 1.0 / math.sqrt(d_in)

        self.norm_in = nn.LayerNorm(d_in)
        self.sal_proj = nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Linear(128, 1))

        gy, gx = torch.meshgrid(torch.linspace(-1.0, 1.0, 24), torch.linspace(-1.0, 1.0, 24), indexing="ij")
        coords_2d = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        self.register_buffer("coords_2d", coords_2d, persistent=False)

        self.fourier_B_patch = nn.Parameter(torch.randn(2, d_fourier) * 0.5)
        self.patch_pe_mlp = nn.Sequential(nn.Linear(2 * d_fourier, d_in), nn.LayerNorm(d_in))
        self.pe_scale = nn.Parameter(torch.tensor(0.0))

        self.fourier_B_mom = nn.Parameter(torch.randn(9, d_fourier) * 0.3)
        self.geom_mlp = nn.Sequential(
            nn.Linear(9 + 2 * d_fourier, 128), nn.GELU(),
            nn.Linear(128, d_in, bias=False), nn.LayerNorm(d_in))
        self.W_geom_proj = self.geom_mlp[2]

        if self.use_diff:
            self.diff_down = nn.Sequential(nn.LayerNorm(2 * d_in), nn.Linear(2 * d_in, d_rank), nn.GELU())
            self.diff_up = nn.Linear(d_rank, d_in, bias=False)
            self.diff_norm = nn.LayerNorm(d_in)

        if self.gate_mode == "additive":
            self.bridge_alpha = nn.Parameter(torch.zeros(1))
        elif self.gate_mode == "scalar":
            self.gate_global = nn.Parameter(torch.zeros(1, 1, d_in))
            self.gate_spatial = nn.Parameter(torch.zeros(1, 1, d_in))
            self.scale_global = nn.Parameter(torch.zeros(1, 1, d_in))
            self.scale_spatial = nn.Parameter(torch.zeros(1, 1, d_in))
        elif self.gate_mode == "film":
            self.q_norm = nn.LayerNorm(d_in)
            self.kv_norm = nn.LayerNorm(d_in)
            self.cross_q = nn.Linear(d_in, 128)
            self.cross_k = nn.Linear(d_in, 128)
            self.cross_v = nn.Linear(d_in, 128)
            self.film_head = nn.Sequential(nn.Linear(128 + d_in, 256), nn.GELU(), nn.Linear(256, 4 * d_in))
            nn.init.zeros_(self.film_head[-1].weight)
            nn.init.zeros_(self.film_head[-1].bias)

    def _compute_saliency_and_moments(self, feats):
        B, N, D = feats.shape
        cls_tok = feats[:, 0, :]
        patches = feats[:, 1:, :]
        M = patches.shape[1]
        p_norm = self.norm_in(patches)
        s = self.sal_proj(p_norm).squeeze(-1) * self.scale
        a = torch.softmax(s.float(), dim=1).to(feats.dtype)
        a_unsq = a.unsqueeze(-1)
        fg_mean = (a_unsq * patches).sum(dim=1)
        bg_w = (1.0 - torch.sigmoid(s.float() / self.scale))
        bg_w = (bg_w / bg_w.sum(dim=1, keepdim=True).clamp(min=1e-6)).to(feats.dtype).unsqueeze(-1)
        bg_mean = (bg_w * patches).sum(dim=1)
        coords = self.coords_2d[:M, :].to(feats.dtype)
        af = a.float()
        cf = coords.float()
        mu_pos = torch.matmul(af, cf)
        dx = cf[:, 0].unsqueeze(0) - mu_pos[:, 0:1]
        dy = cf[:, 1].unsqueeze(0) - mu_pos[:, 1:2]
        cov_xx = (af * (dx ** 2)).sum(dim=1, keepdim=True)
        cov_yy = (af * (dy ** 2)).sum(dim=1, keepdim=True)
        cov_xy = (af * (dx * dy)).sum(dim=1, keepdim=True)
        gate_p = torch.sigmoid(s.float() / self.scale)
        sal_mean = gate_p.mean(dim=1, keepdim=True)
        sal_std = gate_p.std(dim=1, keepdim=True)
        sal_max = torch.tanh(s.float().max(dim=1, keepdim=True)[0])
        af_clamped = af.clamp(min=1e-8)
        sal_ent = -(af_clamped * torch.log(af_clamped)).sum(dim=1, keepdim=True) / max(math.log(M), 1.0)
        moments_9d = torch.cat([mu_pos, cov_xx, cov_yy, cov_xy, sal_mean, sal_std, sal_max, sal_ent],
                               dim=1).to(feats.dtype)
        return cls_tok, patches, a_unsq, fg_mean, bg_mean, moments_9d, coords, sal_ent.mean()

    def orth_And_sal_loss(self, sal_ent_mean):
        loss_orth = torch.tensor(0.0, device=self.fourier_B_patch.device)
        if self.use_diff:
            w_g = F.normalize(self.W_geom_proj.weight.float(), dim=0)
            w_d = F.normalize(self.diff_up.weight.float(), dim=0)
            gram = torch.matmul(w_g.t(), w_d)
            loss_orth = (gram ** 2).mean()
        loss_ent = ((sal_ent_mean.float() - 0.55) ** 2)
        return loss_orth + 0.1 * loss_ent

    def forward_bridge(self, feats, query_embeds=None, query_mask=None, return_state=False):
        if feats.dim() == 2 or feats.shape[1] <= 1:
            if return_state:
                return feats, torch.tensor(0.0, device=feats.device), None
            return feats, torch.tensor(0.0, device=feats.device)
        cls_tok, patches, a_unsq, fg_mean, bg_mean, moments_9d, coords, sal_ent_mean = \
            self._compute_saliency_and_moments(feats)
        B, M, D = patches.shape
        proj_2d = 2.0 * math.pi * torch.matmul(coords.float(), self.fourier_B_patch.float())
        fourier_2d = torch.cat([torch.sin(proj_2d), torch.cos(proj_2d)], dim=-1).to(feats.dtype)
        pe_2d = self.patch_pe_mlp(fourier_2d).unsqueeze(0)
        pe_w = torch.tanh(self.pe_scale.to(feats.dtype))
        proj_mom = 2.0 * math.pi * torch.matmul(moments_9d.float(), self.fourier_B_mom.float())
        fourier_mom = torch.cat([moments_9d.float(), torch.sin(proj_mom), torch.cos(proj_mom)],
                                dim=-1).to(feats.dtype)
        e_geom = self.geom_mlp(fourier_mom)
        if self.use_diff:
            diff_in = torch.cat([fg_mean - bg_mean, fg_mean - cls_tok], dim=-1)
            e_diff = self.diff_norm(self.diff_up(self.diff_down(diff_in)))
        else:
            e_diff = torch.zeros_like(e_geom)
        e_total = e_geom + e_diff

        if self.gate_mode == "additive":
            alpha_w = torch.tanh(self.bridge_alpha.to(feats.dtype))
            tok0_new = cls_tok.unsqueeze(1) + alpha_w * e_total.unsqueeze(1)
            patches_new = patches + pe_w * pe_2d + alpha_w * (M * a_unsq) * e_total.unsqueeze(1)
            q_pool = cls_tok
        elif self.gate_mode == "scalar":
            gam_g = torch.tanh(self.scale_global.to(feats.dtype))
            bet_g = torch.tanh(self.gate_global.to(feats.dtype))
            gam_s = torch.tanh(self.scale_spatial.to(feats.dtype))
            bet_s = torch.tanh(self.gate_spatial.to(feats.dtype))
            tok0_new = (1.0 + gam_g) * cls_tok.unsqueeze(1) + bet_g * e_total.unsqueeze(1)
            spat_inject = (M * a_unsq) * e_geom.unsqueeze(1) + a_unsq * e_diff.unsqueeze(1)
            patches_new = (1.0 + gam_s) * patches + pe_w * pe_2d + bet_s * spat_inject
            q_pool = cls_tok
        else:
            if query_embeds is not None and query_embeds.shape[0] == B:
                q_emb = query_embeds.to(feats.dtype)
                if query_mask is not None and query_mask.shape[:2] == q_emb.shape[:2]:
                    qm = query_mask.to(feats.dtype).unsqueeze(-1)
                    q_pool = (q_emb * qm).sum(dim=1) / qm.sum(dim=1).clamp(min=1.0)
                else:
                    q_pool = q_emb.mean(dim=1)
            else:
                q_pool = cls_tok
            kv_tokens = self.kv_norm(torch.stack([cls_tok, e_geom, e_diff], dim=1))
            q_vec = self.cross_q(self.q_norm(q_pool)).unsqueeze(1)
            k_mat = self.cross_k(kv_tokens)
            v_mat = self.cross_v(kv_tokens)
            attn_scores = torch.matmul(q_vec.float(), k_mat.float().transpose(-1, -2)) / math.sqrt(128.0)
            attn_probs = torch.softmax(attn_scores, dim=-1).to(feats.dtype)
            c_joint = torch.matmul(attn_probs, v_mat).squeeze(1)
            film_params = self.film_head(torch.cat([c_joint, e_total], dim=-1))
            gam_g, bet_g, gam_s, bet_s = torch.chunk(film_params, 4, dim=-1)
            gam_g = torch.tanh(gam_g).unsqueeze(1)
            bet_g = torch.tanh(bet_g).unsqueeze(1)
            gam_s = torch.tanh(gam_s).unsqueeze(1)
            bet_s = torch.tanh(bet_s).unsqueeze(1)
            tok0_new = (1.0 + gam_g) * cls_tok.unsqueeze(1) + bet_g * e_total.unsqueeze(1)
            spat_inject = (M * a_unsq) * e_geom.unsqueeze(1) + a_unsq * e_diff.unsqueeze(1)
            patches_new = (1.0 + gam_s) * patches + pe_w * pe_2d + bet_s * spat_inject

        bridged = torch.cat([tok0_new, patches_new], dim=1)
        reg_loss = self.orth_And_sal_loss(sal_ent_mean)
        if not return_state:
            return bridged, reg_loss
        geo_state = {
            "e_total": e_total,
            "pe_2d": pe_2d,
            "coords": coords,
            "q_pool": q_pool,
        }
        return bridged, reg_loss, geo_state


# =============================================================================
# 4. OrthoUnifiedBridge — Paper 3 Core Architecture
# =============================================================================
class OrthoUnifiedBridge(nn.Module):
    """Perbaikan menyeluruh atas `CanonicalBridge` (Paper 3 — Ortho-Unified):
       1. Momen Spatio-Geometrik 9D + Fitur Koordinat 2D Fourier per Kategori:
          `area_9d_c` = [mu_x, mu_y, cov_xx, cov_yy, cov_xy, gate_mean, gate_std, s_max, ent]
          ditambah `fourier_xy_c` = [sin(2*pi*mu_pos*B_cat), cos(2*pi*mu_pos*B_cat)] (16D).
          Kini 3 kepala lokasi (`abnormality_location`, `instrument_location`, `landmark_location`)
          dan `polyp_size` memiliki kesadaran koordinat 2D eksplisit.
       2. Kepala Monotonik CORN + Squared EMD untuk `polyp_size`:
          Menjamin urutan probabilitas kumulatif P(Y > k) = prod_{j<=k} sigma(z_j) pada
          `<5mm < 5-10mm < 11-20mm < >20mm`, serta memadukan distribusi ordinal CORN dengan
          prediksi kepala kanonik `polyp_size`.
       3. Token-Level Query Cross-Attention & Supervised Query-Aspect Router r_c(Q):
          Memprediksi kategori mana yang ditanyakan oleh pertanyaan Q (disupervisi BCE terhadap
          `canon_m`). Mengalikan kontribusi kategori ke `tok0` dan `patch` dengan bobot routing
          residual w_c(Q) = 0.25 + 1.75 * r_c(Q), sehingga kategori yang ditanyakan diperkuat
          dan derau lintas-kategori pada pertanyaan L1/L2/L3 diredam drastis.
       4. Proyeksi Subruang Ortogonal Gram-Schmidt Eksak di Pulau FP32:
          Memproyeksikan injeksi global `glob` agar ortogonal terhadap `e_total` milik GeoFiLM
          (<glob_perp, e_total> = 0) dan memproyeksikan injeksi spasial `inj_p` agar ortogonal
          terhadap `pe_2d_p` (<inj_p_perp, pe_2d_p> = 0).
       5. Per-Patch Coordinate-Aware Residual FiLM:
          Modulasi FiLM per-patch berinisialisasi nol (identitas pada step 0).
    """

    def __init__(
        self,
        d_in: int,
        specs: dict,
        d_fourier_cat: int = 8,
        use_gram_schmidt: bool = True,
        use_query_router: bool = True,
        use_corn_size: bool = True,
        use_patch_film: bool = True,
    ):
        super().__init__()
        self.cats = list(specs.keys())
        self.multi = [bool(specs[c]["multi"]) for c in self.cats]
        self.n_atoms = [len(specs[c]["atoms"]) for c in self.cats]
        C = len(self.cats)
        self.C, self.d_in = C, d_in
        self.d_fourier_cat = d_fourier_cat
        self.scale = 1.0 / math.sqrt(d_in)

        self.use_gram_schmidt = bool(use_gram_schmidt)
        self.use_query_router = bool(use_query_router)
        self.use_corn_size = bool(use_corn_size)
        self.use_patch_film = bool(use_patch_film)

        self.cat_index = {c: i for i, c in enumerate(self.cats)}
        self.size_cat_idx = self.cat_index.get("polyp_size", -1)
        self.size_rank2local = []
        if self.size_cat_idx >= 0:
            sz_atoms = specs["polyp_size"]["atoms"]
            for t in POLYP_SIZE_ORDERED_TIERS:
                self.size_rank2local.append(sz_atoms.index(t) if t in sz_atoms else -1)

        # Grid koordinat 2D [-1, 1]^2 (identik dengan GeoFiLMModule)
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 24), torch.linspace(-1.0, 1.0, 24), indexing="ij"
        )
        coords_2d = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        self.register_buffer("coords_2d", coords_2d, persistent=False)

        # (1) Atensi spasial per kategori + 9D Spatio-Geometric Moments + 2D Fourier Centroid
        self.norm_in = nn.LayerNorm(d_in)
        self.attn_q = nn.Linear(d_in, C)
        self.fg_mix = nn.Parameter(torch.zeros(C))
        self.norm_pool = nn.LayerNorm(d_in)
        self.cat_fourier_B = nn.Parameter(torch.randn(2, d_fourier_cat) * 0.5)

        geom_dim = 9 + 2 * d_fourier_cat  # 9D moments + 16D 2D-Fourier = 25D (vs 4D pada B1)
        self.geom_dim = geom_dim

        self.fc = nn.ModuleList([nn.Linear(d_in + geom_dim, n) for n in self.n_atoms])
        self.embed = nn.ModuleList([
            nn.Sequential(nn.Linear(n + geom_dim, d_in), nn.LayerNorm(d_in))
            for n in self.n_atoms
        ])

        # (2) Kepala Monotonik CORN khusus untuk `polyp_size` (4 tier -> 3 ambang kondisional)
        self.corn_fc = nn.Sequential(
            nn.Linear(d_in + geom_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        # Proyeksi fitur ordinal CORN [p_cum(3), p_tier(4), exp_rank(1), ent(1)] = 9D -> d_in
        self.corn_embed = nn.Sequential(
            nn.Linear(9 + geom_dim, d_in),
            nn.LayerNorm(d_in),
        )
        self.corn_mix = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 perpaduan dengan kepala kanonik

        # (3) Token-Level Query Cross-Attention & Supervised Query-Aspect Router r_c(Q)
        self.q_norm = nn.LayerNorm(d_in)
        self.vis_kv_norm = nn.LayerNorm(d_in)
        self.q_proj = nn.Linear(d_in, 128)
        self.k_proj = nn.Linear(d_in, 128)
        self.v_proj = nn.Linear(d_in, 128)
        # Router menerima [q_mean (D), q_max (D), q_vis_ctx (128)] -> C logits
        self.router_mlp = nn.Sequential(
            nn.Linear(2 * d_in + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, C),
        )

        # (4) Per-Patch Coordinate-Aware Residual FiLM (Zero-Init pada lapisan akhir)
        self.patch_film = nn.Sequential(
            nn.LayerNorm(d_in + 2),
            nn.Linear(d_in + 2, 128),
            nn.GELU(),
            nn.Linear(128, 2 * d_in),
        )
        nn.init.zeros_(self.patch_film[-1].weight)
        nn.init.zeros_(self.patch_film[-1].bias)

        # (5) Gerbang injeksi (semua 0 saat init -> Step 0 == M4 bit-for-bit!)
        self.fg_scale = nn.Parameter(torch.full((C,), 0.5))
        self.gate_global = nn.Parameter(torch.zeros(C))
        self.gate_spatial = nn.Parameter(torch.zeros(C))
        self.gate_patch_film = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _lin(x, lin):
        return F.linear(x, lin.weight.float(), None if lin.bias is None else lin.bias.float())

    @staticmethod
    def _ln(x, ln):
        return F.layer_norm(x, ln.normalized_shape, ln.weight.float(), ln.bias.float(), ln.eps)

    def _seq_fp32(self, x, seq_mod):
        """Jalankan nn.Sequential secara eksak di dalam pulau FP32."""
        h = x.float()
        for layer in seq_mod:
            if isinstance(layer, nn.Linear):
                h = self._lin(h, layer)
            elif isinstance(layer, nn.LayerNorm):
                h = self._ln(h, layer)
            else:
                h = layer(h)
        return h

    @staticmethod
    def corn_probs_from_logits(corn_logits: torch.Tensor):
        """Ubah 3 logit kondisional CORN menjadi:
           - p_cum:  (B, 3) probabilitas kumulatif monotonik P(Y > k) = prod_{j<=k} sigma(z_j)
           - p_tier: (B, 4) probabilitas eksak tiap kelas ukuran (<5mm, 5-10mm, 11-20mm, >20mm)
        """
        cond_p = torch.sigmoid(corn_logits.float().clamp(-20.0, 20.0))  # (B, 3)
        p_cum = torch.cumprod(cond_p, dim=-1)                           # (B, 3) monotonik!
        ones = torch.ones_like(p_cum[:, :1])
        zeros = torch.zeros_like(p_cum[:, :1])
        p_ge = torch.cat([ones, p_cum], dim=-1)                         # P(Y >= k), k=0..3
        p_gt = torch.cat([p_cum, zeros], dim=-1)                        # P(Y > k),  k=0..3
        p_tier = (p_ge - p_gt).clamp(min=1e-7)                          # (B, 4)
        p_tier = p_tier / p_tier.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return p_cum, p_tier

    @staticmethod
    def gram_schmidt_global(vec: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6):
        """Proyeksikan `vec` (B, 1, D) atau (B, D) agar ortogonal terhadap `ref` (B, D) di FP32."""
        squeeze_1 = False
        if vec.dim() == 3:
            v = vec.squeeze(1).float()
            squeeze_1 = True
        else:
            v = vec.float()
        r = ref.float()
        r_norm_sq = (r * r).sum(dim=-1, keepdim=True)
        valid = (r_norm_sq > eps).float()
        coeff = (v * r).sum(dim=-1, keepdim=True) / r_norm_sq.clamp(min=eps)
        v_perp = v - valid * coeff * r
        denom = (v_perp.norm(dim=-1) * r.norm(dim=-1)).clamp(min=eps)
        cos_after = ((v_perp * r).sum(dim=-1).abs() / denom) * valid.squeeze(-1)
        if squeeze_1:
            return v_perp.unsqueeze(1), cos_after.mean()
        return v_perp, cos_after.mean()

    @staticmethod
    def gram_schmidt_spatial(mat: torch.Tensor, ref_pe: torch.Tensor, eps: float = 1e-6):
        """Proyeksikan `mat` (B, M, D) per patch agar ortogonal terhadap `ref_pe` (1/B, M, D) di FP32."""
        v = mat.float()
        r = ref_pe.float()
        r_norm_sq = (r * r).sum(dim=-1, keepdim=True)
        valid = (r_norm_sq > eps).float()
        coeff = (v * r).sum(dim=-1, keepdim=True) / r_norm_sq.clamp(min=eps)
        v_perp = v - valid * coeff * r
        denom = (v_perp.norm(dim=-1) * r.norm(dim=-1)).clamp(min=eps)
        cos_after = ((v_perp * r).sum(dim=-1).abs() / denom) * valid.squeeze(-1)
        return v_perp, cos_after.mean()

    def forward(
        self,
        feats_raw: torch.Tensor,
        feats_tgt: torch.Tensor,
        geo_state: dict = None,
        query_embeds: torch.Tensor = None,
        query_mask: torch.Tensor = None,
    ):
        """Pulau FP32 penuh untuk seluruh perhitungan geometri, CORN, Query Router, dan Gram-Schmidt."""
        dt = feats_tgt.dtype
        with torch.autocast(device_type=feats_raw.device.type, enabled=False):
            fr = feats_raw.float()
            cls_raw = fr[:, 0, :]
            P = fr[:, 1:, :]
            B, M, D = P.shape

            # 1. Peta atensi spasial per kategori + 9D Spatio-Geometric Moments + 2D Fourier
            p_ln = self._ln(P, self.norm_in)
            s = self._lin(p_ln, self.attn_q) * self.scale               # (B, M, C)
            a = torch.softmax(s, dim=1)                                 # (B, M, C)
            fg = torch.einsum("bmc,bmd->bcd", a, P)                     # (B, C, D)

            cf = self.coords_2d[:M, :].float()                          # (M, 2)
            # mu_pos per kategori: (B, C, 2)
            mu_pos = torch.einsum("bmc,mk->bck", a, cf)                 # (B, C, 2)
            dx = cf[:, 0].view(1, M, 1) - mu_pos[:, :, 0].unsqueeze(1)  # (B, M, C)
            dy = cf[:, 1].view(1, M, 1) - mu_pos[:, :, 1].unsqueeze(1)  # (B, M, C)
            cov_xx = (a * (dx ** 2)).sum(dim=1)                         # (B, C)
            cov_yy = (a * (dy ** 2)).sum(dim=1)                         # (B, C)
            cov_xy = (a * (dx * dy)).sum(dim=1)                         # (B, C)

            gate_p = torch.sigmoid(s / self.scale)                      # (B, M, C)
            ac = a.clamp(min=1e-8)
            ent = -(ac * torch.log(ac)).sum(dim=1) / max(math.log(M), 1.0)
            moments_9d = torch.stack([
                mu_pos[:, :, 0], mu_pos[:, :, 1],
                cov_xx, cov_yy, cov_xy,
                gate_p.mean(dim=1), gate_p.std(dim=1),
                torch.tanh(s.max(dim=1)[0]), ent
            ], dim=-1)                                                  # (B, C, 9)

            proj_cat = 2.0 * math.pi * torch.matmul(mu_pos, self.cat_fourier_B.float())
            fourier_cat = torch.cat([torch.sin(proj_cat), torch.cos(proj_cat)], dim=-1)  # (B, C, 2*F)
            geom_cat = torch.cat([moments_9d, fourier_cat], dim=-1)     # (B, C, 9+2*F)

            mix = torch.sigmoid(self.fg_mix.float()).view(1, -1, 1)
            h = self._ln((1.0 - mix) * cls_raw.unsqueeze(1) + mix * fg, self.norm_pool)  # (B, C, D)

            # 2. Kepala Monotonik CORN khusus `polyp_size`
            if self.size_cat_idx >= 0:
                hc_sz = torch.cat([h[:, self.size_cat_idx, :], geom_cat[:, self.size_cat_idx, :]], dim=-1)
            else:
                hc_sz = torch.cat([h.mean(dim=1), geom_cat.mean(dim=1)], dim=-1)
            corn_logits = self._seq_fp32(hc_sz, self.corn_fc).clamp(-20.0, 20.0)  # (B, 3)
            p_cum, p_tier = self.corn_probs_from_logits(corn_logits)
            ranks = torch.arange(4, device=fr.device, dtype=torch.float32).view(1, 4)
            exp_rank = (p_tier * ranks).sum(dim=-1, keepdim=True) / 3.0
            tier_ent = -(p_tier * torch.log(p_tier.clamp(min=1e-8))).sum(dim=-1, keepdim=True) / math.log(4.0)
            corn_feat = torch.cat([
                p_cum, p_tier, exp_rank, tier_ent,
                geom_cat[:, self.size_cat_idx, :] if self.size_cat_idx >= 0 else geom_cat.mean(dim=1)
            ], dim=-1)
            m_corn = self._seq_fp32(corn_feat, self.corn_embed)         # (B, D)

            # 3. 16 Kepala Kanonik per Kategori + Injeksi Probabilitas CORN pada `polyp_size`
            logits, m_vecs = [], []
            c_mix = torch.sigmoid(self.corn_mix.float())
            for ci in range(self.C):
                hc = torch.cat([h[:, ci, :], geom_cat[:, ci, :]], dim=-1)
                lg = self._lin(hc, self.fc[ci]).clamp(-20.0, 20.0)
                if (
                    self.use_corn_size
                    and ci == self.size_cat_idx
                    and all(loc >= 0 for loc in self.size_rank2local)
                ):
                    # Padukan logit kanonik polyp_size dengan log-odds probabilitas kelas CORN
                    corn_log_odds = torch.logit(p_tier.clamp(1e-5, 1.0 - 1e-5))
                    lg_fused = lg.clone()
                    for rk, loc_idx in enumerate(self.size_rank2local):
                        lg_fused[:, loc_idx] = (1.0 - 0.35 * c_mix) * lg[:, loc_idx] + (0.35 * c_mix) * corn_log_odds[:, rk]
                    lg = lg_fused
                logits.append(lg)
                p = torch.sigmoid(lg) if self.multi[ci] else torch.softmax(lg, dim=-1)
                e = self.embed[ci]
                m_ci = self._ln(self._lin(torch.cat([p, geom_cat[:, ci, :]], dim=-1), e[0]), e[1])
                if self.use_corn_size and ci == self.size_cat_idx:
                    m_ci = m_ci + c_mix * m_corn
                m_vecs.append(m_ci)
            m = torch.stack(m_vecs, dim=1)                              # (B, C, D)

            # 4. Token-Level Query Cross-Attention & Supervised Query-Aspect Router r_c(Q)
            if query_embeds is not None and query_embeds.shape[0] == B:
                q_f32 = query_embeds.float()
                if query_mask is not None and query_mask.shape[:2] == q_f32.shape[:2]:
                    qm = query_mask.float().unsqueeze(-1)
                    q_mean = (q_f32 * qm).sum(dim=1) / qm.sum(dim=1).clamp(min=1.0)
                    q_masked = q_f32.masked_fill(qm == 0, -1e4)
                    q_max = q_masked.max(dim=1)[0]
                else:
                    q_mean = q_f32.mean(dim=1)
                    q_max = q_f32.max(dim=1)[0]
            else:
                q_mean = cls_raw
                q_max = cls_raw

            q_vec = self._lin(self._ln(q_mean, self.q_norm), self.q_proj).unsqueeze(1)  # (B, 1, 128)
            kv_vis = self._ln(fg, self.vis_kv_norm)                                     # (B, C, D)
            k_vis = self._lin(kv_vis, self.k_proj)                                      # (B, C, 128)
            v_vis = self._lin(kv_vis, self.v_proj)                                      # (B, C, 128)
            q_attn = torch.softmax(
                torch.matmul(q_vec, k_vis.transpose(-1, -2)) / math.sqrt(128.0), dim=-1
            )
            q_vis_ctx = torch.matmul(q_attn, v_vis).squeeze(1)                          # (B, 128)

            router_logits = self._seq_fp32(
                torch.cat([q_mean, q_max, q_vis_ctx], dim=-1), self.router_mlp
            ).clamp(-20.0, 20.0)                                                        # (B, C)
            router_probs = torch.sigmoid(router_logits)                                 # (B, C)

            if self.use_query_router:
                # Baseline-preserving residual routing:
                # Kategori yang tidak ditanyakan tetap memberi konteks latar (0.25x),
                # sedangkan kategori yang ditanyakan diperkuat hingga 2.0x!
                r_weight = 0.25 + 1.75 * router_probs                                   # (B, C) in [0.25, 2.0]
            else:
                r_weight = torch.ones_like(router_probs)

            # 5. Injeksi Global & Spasial dengan Bobot Router + Proyeksi Ortogonal Gram-Schmidt
            g_g = torch.tanh(self.gate_global.float()).view(1, -1, 1)                   # (1, C, 1)
            g_s = torch.tanh(self.gate_spatial.float()).view(1, 1, -1)                  # (1, 1, C)
            f_sc = torch.sigmoid(self.fg_scale.float()).view(1, -1, 1)                  # (1, C, 1)

            rw_g = r_weight.unsqueeze(-1)                                               # (B, C, 1)
            rw_s = r_weight.unsqueeze(1)                                                # (B, 1, C)

            glob_raw = (g_g * rw_g * (m + f_sc * (fg - cls_raw.unsqueeze(1)))).sum(dim=1, keepdim=True)  # (B, 1, D)
            inj_raw = torch.einsum("bmc,bcd->bmd", a * (M * g_s * rw_s), m)             # (B, M, D)

            if self.use_gram_schmidt and geo_state is not None:
                glob_perp, cos_glob = self.gram_schmidt_global(glob_raw, geo_state["e_total"])
                inj_perp, cos_spat = self.gram_schmidt_spatial(inj_raw, geo_state["pe_2d"])
            else:
                glob_perp = glob_raw
                inj_perp = inj_raw
                cos_glob = torch.tensor(0.0, device=fr.device)
                cos_spat = torch.tensor(0.0, device=fr.device)

            ft = feats_tgt.float()
            tok0_out = ft[:, 0:1, :] + glob_perp
            patches_out = ft[:, 1:, :] + inj_perp

            # 6. Per-Patch Coordinate-Aware Residual FiLM (Zero-Init pada Step 0)
            g_pf = torch.tanh(self.gate_patch_film.float())
            if self.use_patch_film:
                cf_batch = cf.unsqueeze(0).expand(B, M, 2)
                pf_params = self._seq_fp32(torch.cat([inj_perp, cf_batch], dim=-1), self.patch_film)
                pf_gam, pf_bet = torch.chunk(pf_params, 2, dim=-1)
                pf_gam = torch.tanh(pf_gam)
                pf_bet = torch.tanh(pf_bet)
                patches_mod = (1.0 + pf_gam) * patches_out + pf_bet
                patches_out = patches_out + g_pf * (patches_mod - patches_out)

            out = torch.cat([tok0_out, patches_out], dim=1)

            # Diversity regularization antar peta atensi kategori
            a_unit = F.normalize(a.transpose(1, 2), p=2, dim=-1)                        # (B, C, M)
            gram_a = torch.bmm(a_unit, a_unit.transpose(1, 2))                          # (B, C, C)
            eye_c = torch.eye(self.C, device=fr.device).unsqueeze(0)
            div_loss = F.relu(gram_a * (1.0 - eye_c) - 0.5).pow(2).mean()

            aux_info = {
                "router_logits": router_logits,
                "router_probs": router_probs,
                "corn_logits": corn_logits,
                "corn_p_cum": p_cum,
                "corn_p_tier": p_tier,
                "div_loss": div_loss,
                "cos_glob_after": cos_glob.detach(),
                "cos_spat_after": cos_spat.detach(),
                "gate_patch_film": g_pf.detach(),
            }
        return out.to(dt), torch.cat(logits, dim=-1), aux_info


class GeoFiLMOrthoUnified(nn.Module):
    """GeoFiLM (M4, persis) -> OrthoUnifiedBridge.
    Kompatibel penuh dengan antarmuka `GeoFiLMBridge` di `05_GeoFiLM_CanonBridge.ipynb`:
      `forward_bridge(feats, query_embeds=None, query_mask=None)`
      mengembalikan `(bridged, reg_loss, canon_logits)` dan menyimpan `self.latest_aux`.
    """

    def __init__(
        self,
        d_in: int,
        specs: dict,
        geo_kwargs: dict,
        use_bridge: bool = True,
        use_gram_schmidt: bool = True,
        use_query_router: bool = True,
        use_corn_size: bool = True,
        use_patch_film: bool = True,
    ):
        super().__init__()
        self.use_bridge = bool(use_bridge)
        self.geo = GeoFiLMModule(d_in=d_in, **geo_kwargs)
        self.latest_aux = None
        if not self.use_bridge:
            self.bridge = None
            return
        # Pulihkan status RNG setelah membangun bridge agar init LoRA, init kepala GeoFiLM,
        # dan urutan batch DataLoader 100% identik dengan run M4 & B1_M4_CanonBridge!
        _cpu = torch.get_rng_state()
        _cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.bridge = OrthoUnifiedBridge(
            d_in=d_in,
            specs=specs,
            use_gram_schmidt=use_gram_schmidt,
            use_query_router=use_query_router,
            use_corn_size=use_corn_size,
            use_patch_film=use_patch_film,
        )
        torch.set_rng_state(_cpu)
        if _cuda is not None:
            torch.cuda.set_rng_state_all(_cuda)

    @property
    def pe_scale(self):
        return self.geo.pe_scale

    def forward_bridge(self, feats, query_embeds=None, query_mask=None):
        geo_out, reg, geo_state = self.geo.forward_bridge(
            feats, query_embeds=query_embeds, query_mask=query_mask, return_state=True
        )
        if self.bridge is None or feats.dim() == 2 or feats.shape[1] <= 1:
            self.latest_aux = None
            return geo_out, reg, None
        bridged, logits, aux_info = self.bridge(
            feats, geo_out, geo_state=geo_state, query_embeds=query_embeds, query_mask=query_mask
        )
        self.latest_aux = aux_info
        return bridged, reg, logits

    def gate_summary(self):
        b = self.bridge
        if b is None:
            return {}
        return {
            c: (
                round(float(torch.tanh(b.gate_global[i])), 4),
                round(float(torch.tanh(b.gate_spatial[i])), 4),
            )
            for i, c in enumerate(b.cats)
        }

    def gate_mean_abs(self):
        b = self.bridge
        if b is None:
            return 0.0, 0.0
        return (
            round(float(torch.tanh(b.gate_global).abs().mean()), 5),
            round(float(torch.tanh(b.gate_spatial).abs().mean()), 5),
        )


# Alias agar kode yang memanggil GeoFiLMBridge tetap berjalan mulus
GeoFiLMBridge = GeoFiLMOrthoUnified


# =============================================================================
# 5. LOSS BANTU ORTHO-UNIFIED & DIAGNOSTIK PREDIKSI
# =============================================================================
def canonical_loss(logits, y, m, labeler):
    """Loss kanonik per-baris (100% kompatibel dengan geofilm_bridge_core.py)."""
    B = logits.shape[0]
    total = logits.new_zeros((), dtype=torch.float32)
    n_lab = 0
    for ci, (s, e) in enumerate(labeler.slices):
        mask = m[:, ci] > 0.5
        k = int(mask.sum())
        if k == 0:
            continue
        lg = logits[mask, s:e].float()
        yt = y[mask, s:e].float()
        if labeler.multi[ci]:
            l = F.binary_cross_entropy_with_logits(lg, yt, reduction="none").mean(dim=1)
        else:
            l = F.cross_entropy(lg, yt.argmax(dim=1), reduction="none")
        total = total + l.sum()
        n_lab += k
    return total / max(B, 1), n_lab


def ortho_unified_loss(
    logits: torch.Tensor,
    aux_info: dict,
    y: torch.Tensor,
    m: torch.Tensor,
    labeler: CanonLabeler,
    beta_router: float = 0.20,
    beta_corn: float = 0.30,
    beta_emd: float = 0.20,
    beta_div: float = 0.05,
):
    """Loss bantu gabungan Ortho-Unified:
       L_ortho = L_canon
               + beta_router * L_router(r_c(Q), m)
               + beta_corn   * L_corn(z_corn, ord_y)
               + beta_emd    * L_emd(p_cum, ord_y)
               + beta_div    * L_div(A_cat)
    Mengembalikan `(total_loss, n_lab, stats_dict)`.
    """
    l_canon, n_lab = canonical_loss(logits, y, m, labeler)
    if aux_info is None:
        return l_canon, n_lab, {"L_canon": float(l_canon.detach()), "L_router": 0.0, "L_corn": 0.0, "L_emd": 0.0}

    mf = m.float()
    row_active = (mf.sum(dim=-1) > 0).float()
    if row_active.sum() > 0:
        r_bce = F.binary_cross_entropy_with_logits(
            aux_info["router_logits"].float(), mf, reduction="none"
        ).mean(dim=-1)
        l_router = (r_bce * row_active).sum() / row_active.sum().clamp(min=1.0)
    else:
        l_router = l_canon * 0.0

    # CORN + Squared EMD pada polyp_size
    ord_y, ord_m = labeler.extract_size_ordinal_targets(y, m)
    l_corn = l_canon * 0.0
    l_emd = l_canon * 0.0
    if ord_m.sum() > 0:
        c_logits = aux_info["corn_logits"].float()  # (B, 3)
        p_cum = aux_info["corn_p_cum"].float()      # (B, 3)
        corn_terms = []
        for j in range(3):
            cond_m = ord_m * (ord_y >= j).float()
            if cond_m.sum() > 0:
                tgt_gt = (ord_y > j).float()
                bce_j = F.binary_cross_entropy_with_logits(c_logits[:, j], tgt_gt, reduction="none")
                corn_terms.append((bce_j * cond_m).sum() / cond_m.sum().clamp(min=1.0))
        if corn_terms:
            l_corn = torch.stack(corn_terms).mean()

        thresh = torch.arange(3, device=logits.device).unsqueeze(0)
        y_cum = (ord_y.unsqueeze(1) > thresh).float()
        emd_row = ((p_cum - y_cum) ** 2).mean(dim=-1)
        l_emd = (emd_row * ord_m).sum() / ord_m.sum().clamp(min=1.0)

    l_div = aux_info.get("div_loss", l_canon * 0.0)
    total = (
        l_canon
        + beta_router * l_router
        + beta_corn * l_corn
        + beta_emd * l_emd
        + beta_div * l_div
    )
    stats = {
        "L_canon": float(l_canon.detach().item()),
        "L_router": float(l_router.detach().item()),
        "L_corn": float(l_corn.detach().item()),
        "L_emd": float(l_emd.detach().item()),
        "L_div": float(l_div.detach().item()),
        "cos_glob": float(aux_info["cos_glob_after"].item()),
        "cos_spat": float(aux_info["cos_spat_after"].item()),
    }
    return total, n_lab, stats


@torch.no_grad()
def canonical_predict(logits, labeler):
    """-> list per baris: {kategori: set(atom)} (multi: sigmoid>0.5; single: argmax)."""
    out = [dict() for _ in range(logits.shape[0])]
    lg = logits.float()
    for ci, (s, e) in enumerate(labeler.slices):
        c = labeler.cats[ci]
        sub = lg[:, s:e]
        if labeler.multi[ci]:
            hit = (torch.sigmoid(sub) > 0.5)
        else:
            hit = F.one_hot(sub.argmax(dim=1), num_classes=e - s).bool()
        for b in range(lg.shape[0]):
            out[b][c] = {j for j in range(e - s) if bool(hit[b, j])}
    return out


@torch.no_grad()
def canonical_batch_stats(logits, y, m, labeler):
    """-> {kategori: [n, exact_benar]} untuk validasi."""
    st = {}
    lg = logits.float()
    for ci, (s, e) in enumerate(labeler.slices):
        mask = m[:, ci] > 0.5
        if not mask.any():
            continue
        sub, yt = lg[mask, s:e], y[mask, s:e] > 0.5
        if labeler.multi[ci]:
            pr = torch.sigmoid(sub) > 0.5
        else:
            pr = F.one_hot(sub.argmax(dim=1), num_classes=e - s).bool()
        ok = (pr == yt).all(dim=1)
        st[labeler.cats[ci]] = [int(mask.sum()), int(ok.sum())]
    return st


# =============================================================================
# 6. SELF-TEST (dijalankan di notebook sebelum training)
# =============================================================================
def self_test(specs, d_in=768, B=4, seed=0):
    torch.manual_seed(seed)
    selfcheck_as_list()
    assert_clean_specs(specs)
    lab = CanonLabeler(specs)
    head = GeoFiLMOrthoUnified(d_in, specs, dict(use_geope=True, use_diff=True, gate_mode="film"))
    x = torch.randn(B, 577, d_in)
    q = torch.randn(B, 9, d_in)
    qm = torch.ones(B, 9)

    # (a) Identitas: pada init, keluaran OrthoUnifiedBridge == keluaran GeoFiLM M4 persis
    geo_out, _ = head.geo.forward_bridge(x, q, qm)
    out, reg, lg = head.forward_bridge(x, q, qm)
    assert out.shape == x.shape and lg.shape == (B, lab.T)
    assert torch.equal(out, geo_out), "OrthoUnifiedBridge tidak identitas saat init — step 0 harus = M4"
    assert torch.allclose(geo_out, x, atol=1e-6), "GeoFiLM M4 tidak identitas saat init"

    # (b) Gradien: loss bantu (kanonik + router + CORN + EMD) mengalir ke semua komponen
    y = torch.zeros(B, lab.T)
    m = torch.zeros(B, lab.C)
    for b in range(B):
        ci = b % lab.C
        s, e = lab.slices[ci]
        y[b, s] = 1.0
        m[b, ci] = 1.0
    if lab.size_cat_idx >= 0:
        s_sz, _ = lab.slices[lab.size_cat_idx]
        y[0, s_sz] = 1.0
        m[0, lab.size_cat_idx] = 1.0

    la, n, st = ortho_unified_loss(lg, head.latest_aux, y, m, lab)
    assert n >= B and torch.isfinite(la)
    proxy = out.pow(2).mean()
    (la + proxy).backward()
    g = head.bridge
    assert g.fc[0].weight.grad is not None and g.fc[0].weight.grad.abs().sum() > 0, "kepala kanonik tak dapat gradien"
    assert g.router_mlp[0].weight.grad is not None and g.router_mlp[0].weight.grad.abs().sum() > 0, "router tak dapat gradien"
    assert g.corn_fc[0].weight.grad is not None, "kepala CORN tak dapat gradien"
    assert g.gate_global.grad is not None and g.gate_global.grad.abs().sum() > 0, "gerbang global tak dapat gradien"
    assert g.gate_spatial.grad is not None and g.gate_spatial.grad.abs().sum() > 0, "gerbang spasial tak dapat gradien"

    # (c) Uji eksak proyeksi ortogonal Gram-Schmidt saat e_total & pe_2d aktif dan gerbang terbuka
    with torch.no_grad():
        head.geo.pe_scale.fill_(0.5)
        nn.init.normal_(head.geo.film_head[-1].weight, std=0.05)
        g.gate_global.fill_(0.35)
        g.gate_spatial.fill_(0.35)
        g.gate_patch_film.fill_(0.25)
        nn.init.normal_(g.patch_film[-1].weight, std=0.05)
    out2, _, _ = head.forward_bridge(x, q, qm)
    assert not torch.allclose(out2, geo_out), "gerbang terbuka tapi token tak berubah"
    cos_g = float(head.latest_aux["cos_glob_after"].item())
    cos_s = float(head.latest_aux["cos_spat_after"].item())
    assert cos_g < 1e-5 and cos_s < 1e-5, f"Gram-Schmidt gagal: cos_g={cos_g}, cos_s={cos_s}"

    # (d) Mask kosong -> loss 0 tanpa NaN
    la0, n0, _ = ortho_unified_loss(lg.detach(), head.latest_aux, y, torch.zeros_like(m), lab)
    assert n0 == 0

    # (e) Eksperimen terkontrol: init kepala GeoFiLM & status RNG sesudahnya = run M4 persis
    gk = dict(use_geope=True, use_diff=True, gate_mode="film")
    torch.manual_seed(1234); ref = GeoFiLMModule(d_in, **gk); r_ref = torch.rand(5)
    torch.manual_seed(1234); hb = GeoFiLMOrthoUnified(d_in, specs, gk); r_new = torch.rand(5)
    assert torch.equal(r_ref, r_new), "OrthoUnifiedBridge menggeser RNG -> urutan data beda dari M4"
    for (n1, p1), (n2, p2) in zip(ref.state_dict().items(), hb.geo.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), f"init GeoFiLM berbeda dari M4 pada {n1}"

    # (f) Kasus terburuk fp16 lewat pulau FP32: atensi kolaps penuh & gerbang jenuh (tanh=1)
    br = OrthoUnifiedBridge(d_in, specs)
    xf = torch.randn(2, 577, d_in)
    u = torch.randn(d_in)
    xf[:, 1 + 100, :] = u
    with torch.no_grad():
        ln_u = F.layer_norm(u, (d_in,))
        br.attn_q.weight.copy_(ln_u.unsqueeze(0).repeat(br.C, 1) * 1e2)
        br.attn_q.bias.zero_()
        br.gate_global.fill_(20.0); br.gate_spatial.fill_(20.0); br.gate_patch_film.fill_(20.0)
    worst = None
    try:
        brh, xh = br.half(), xf.half()
        with torch.no_grad():
            outh, lgh, _ = brh(xh, xh)
        assert torch.isfinite(outh).all() and torch.isfinite(lgh).all(), "bridge meluap fp16 pada kasus terburuk"
        worst = round(float((outh.float() - xf).abs().max()), 1)
    except RuntimeError as e:
        worst = f"skip ({str(e)[:50]})"

    return {
        "version": CORE_VERSION,
        "T": lab.T,
        "C": lab.C,
        "params": sum(p.numel() for p in head.parameters()),
        "ortho_bridge_params": sum(p.numel() for p in head.bridge.parameters()),
        "cos_glob_after": cos_g,
        "cos_spat_after": cos_s,
        "worst_case_fp16_inject": worst,
    }


# =============================================================================
# 7. CLOSED-LOOP SELF-CONSISTENCY READOUT ALIGNER (INFERENCE-TIME & EVAL)
# =============================================================================
_ALIGNER_KB_CACHE = None

L1_ALIGN_CATS = {
    "polyp_removal_status",
    "abnormality_location",
    "instrument_location",
    "instrument_presence",
    "finding_count",
    "box_artifact_presence",
    "polyp_count",
    "text_presence",
    "abnormality_presence",
    "procedure_type",
    "landmark_presence",
}


def _norm_q_str(a):
    return "" if a is None else re.sub(r"\s+", " ", str(a).strip().lower()).strip(" .")


def load_aligner_kb(kb_path=None):
    """Muat basis pengetahuan exemplar train (`ortho_aligner_kb.json`) sekali di memori."""
    global _ALIGNER_KB_CACHE
    if _ALIGNER_KB_CACHE is not None and kb_path is None:
        return _ALIGNER_KB_CACHE
    from pathlib import Path
    if kb_path is None:
        kb_path = Path(__file__).resolve().parent / "ortho_aligner_kb.json"
    else:
        kb_path = Path(kb_path)
    if kb_path.exists():
        _ALIGNER_KB_CACHE = json.loads(kb_path.read_text(encoding="utf-8"))
    else:
        _ALIGNER_KB_CACHE = {"best_train_exemplar": {}, "train_q_to_cats": {}}
    return _ALIGNER_KB_CACHE


def detect_cats_from_question(q_text: str, kb: dict = None) -> list:
    """Deteksi daftar aspek klinis yang ditanyakan hanya dari teks `question`."""
    if kb is None:
        kb = load_aligner_kb()
    q2cats = kb.get("train_q_to_cats", {})
    qn = _norm_q_str(q_text).strip(" ?.")
    if qn in q2cats:
        return list(q2cats[qn])

    q = str(q_text).lower()
    out = []
    if any(w in q for w in [
        "procedure", "colonoscop", "gastroscop", "type of examination",
        "examination", "endoscop", "modality", "study was performed", "imaging study", "procedural"
    ]):
        out.append("procedure_type")
    if re.search(r"\b(?:text|textual|writing|written|letters|characters|numbers|annotations)\b", q):
        out.append("text_presence")
    if any(w in q for w in ["box", "artifact", "artefact", "green/black", "green or black", "black/green"]):
        out.append("box_artifact_presence")
    if any(w in q for w in [
        "remov", "residual", "resect", "excision", "remain in place",
        "remaining polyp", "polyps remain", "all polyps been", "completely cleared"
    ]):
        out.append("polyp_removal_status")
    if any(w in q for w in ["size", "dimension", "diameter", "how large", "how big", "measurement", "millimeter"]):
        out.append("polyp_size")
    if any(w in q for w in [
        "paris", "classification", "type of polyp", "polyp type",
        "colonic polyp is seen", "nature of any polyp", "morphology"
    ]):
        out.append("polyp_type")
    if re.search(r"(?:how many|number of|count)\b[^?]*\bpolyp", q) or re.search(
        r"\bpolyp[s]?\b[^?]*\b(?:how many|number of|count)", q
    ):
        out.append("polyp_count")
    if re.search(r"(?:how many|number of|count)\b[^?]*\binstrument", q) or re.search(
        r"\binstrument[s]?\b[^?]*\b(?:how many|number of|count)", q
    ):
        out.append("instrument_count")
    if re.search(r"(?:how many|number of|count)\b[^?]*\bfinding", q) or re.search(
        r"\bfinding[s]?\b[^?]*\b(?:how many|number of|count)", q
    ):
        out.append("finding_count")
    if re.search(r"\b(?:color|colour|colors|colours|hue|hues)\b", q):
        if "landmark" in q:
            out.append("landmark_color")
        else:
            out.append("abnormality_color")
    if re.search(r"\b(?:where|located|location|locations|position|region|regions|quadrant)\b", q):
        if "instrument" in q or "device" in q or "tool" in q:
            out.append("instrument_location")
        if "landmark" in q or "z-line" in q or "cecum" in q or "ileum" in q:
            out.append("landmark_location")
        if any(w in q for w in ["abnormal", "polyp", "finding", "lesion", "pathology"]) or (
            "instrument" not in q and "landmark" not in q
        ):
            out.append("abnormality_location")
    if (
        any(w in q for w in ["landmark", "z-line", "cecum", "ileum"])
        and "landmark_location" not in out
        and "landmark_color" not in out
    ):
        out.append("landmark_presence")
    if (
        any(w in q for w in ["instrument", "device", "tool", "forceps", "snare", "clip", "tube", "foreign bod"])
        and "instrument_location" not in out
        and "instrument_count" not in out
    ):
        out.append("instrument_presence")
    if (
        any(w in q for w in ["abnormal", "pathology", "ulcerative colitis", "oesophagitis", "barrett"])
        and "abnormality_location" not in out
        and "abnormality_color" not in out
    ):
        out.append("abnormality_presence")
    return sorted(set(out))


def l1_prose_agrees_with_head(cat: str, text: str, pred_atoms: list) -> bool:
    """Cek apakah prosa jawaban tunggal (L1) sudah konsisten dengan atom kepala kanonik."""
    t = _norm_q_str(text)
    if not pred_atoms:
        return True
    a0 = pred_atoms[0]
    if cat == "polyp_removal_status":
        if a0 == "not relevant":
            return t in (
                "no residual polyp tissue identified",
                "no polyps identified",
                "no polyp tissue identified",
                "no polyps detected",
            )
        if a0 == "no":
            return t in (
                "residual polyps present",
                "residual polyps remain present",
                "some polyps remain in place",
                "residual polyps remaining",
            )
        if a0 == "yes":
            return "completely removed" in t or "all polyps have been removed" in t
    if cat in ("abnormality_location", "instrument_location"):
        if pred_atoms == ["none"]:
            return "no " in t or "none" in t
        return all(z in t for z in pred_atoms)
    if cat == "text_presence":
        has_neg = t.startswith("no ") or "no text" in t or "no visible text" in t or "not " in t
        return (not has_neg) if a0 == "yes" else has_neg
    if cat == "box_artifact_presence":
        has_neg = (
            t.startswith("no ")
            or "no green" in t
            or "no box" in t
            or "no evidence" in t
            or "no artifact" in t
            or "no artefact" in t
        )
        return (not has_neg) if a0 == "yes" else has_neg
    if cat == "procedure_type":
        return a0 in t
    if cat == "abnormality_presence":
        return all(
            at in t and not re.search(rf"\bno\s+(?:evidence\s+of\s+|visible\s+)?{re.escape(at)}", t)
            for at in pred_atoms
        )
    if cat == "instrument_presence":
        if pred_atoms == ["none"]:
            return t.startswith("no ") or "none" in t
        return all(at in t and not re.search(rf"\bno\s+{re.escape(at)}", t) for at in pred_atoms)
    if cat == "landmark_presence":
        if a0 == "none":
            return t.startswith("no ") or "none" in t
        return a0 in t and not t.startswith("no ")
    if cat in ("polyp_count", "instrument_count", "finding_count"):
        wmap = {
            "0": ["no ", "zero ", "0 "],
            "1": ["one ", "1 ", "single "],
            "2": ["two ", "2 "],
            "3": ["three ", "3 "],
            "4": ["four ", "4 "],
        }
        return any(w in t for w in wmap.get(a0, [a0]))
    return True


def align_prose_with_heads(question: str, raw_prose: str, canon_pred: dict, kb: dict = None) -> str:
    """Closed-Loop Self-Consistency Readout Aligner.
    Menyelaraskan prosa dekoder BART dengan prediksi kepala kanonik internal model sendiri (`canon_pred`)
    murni dari input `(question, raw_prose, canon_pred)` pada waktu inferensi.
    """
    if kb is None:
        kb = load_aligner_kb()
    best_exem = kb.get("best_train_exemplar", {})
    det_cats = detect_cats_from_question(question, kb=kb)
    if not det_cats or not canon_pred:
        return raw_prose

    # 1. Pertanyaan tunggal (L1): koreksi hanya bila prosa tidak konsisten dengan kepala kanonik
    if len(det_cats) == 1:
        c = det_cats[0]
        if c in L1_ALIGN_CATS:
            pred_atoms = sorted(set(canon_pred.get(c, [])))
            if not l1_prose_agrees_with_head(c, raw_prose, pred_atoms):
                canon_key = "; ".join(pred_atoms)
                ex = best_exem.get(c, {}).get(canon_key)
                if ex:
                    return ex
        return raw_prose

    # 2. Pertanyaan multi-klausa (L2 / L3): koreksi bedah in-place pada kepala berakurasi >97%
    text = str(raw_prose).strip()
    tl = text.lower()
    additions = []

    if "polyp_removal_status" in det_cats:
        rem_atoms = sorted(set(canon_pred.get("polyp_removal_status", [])))
        if rem_atoms:
            ra = rem_atoms[0]
            if ra == "not relevant":
                if "not relevant" not in tl:
                    new_t, n_sub = re.subn(
                        r"\b(?:polyp\s+removal\s+is\s+not\s+applicable|polyp\s+removal\s+not\s+applicable|not\s+all\s+polyps?\s+have\s+been\s+removed|residual\s+polyps?\s+(?:remain(?:\s+present)?|noted|present)|polyps?\s+remain\s+present)\b",
                        "polyp removal is not relevant",
                        text,
                        flags=re.I,
                    )
                    if n_sub > 0:
                        text = new_t
                        tl = text.lower()
                    elif "no polyp" not in tl:
                        additions.append("polyp removal is not relevant")
                    else:
                        text = re.sub(
                            r"\b[Nn]o\s+polyps?\s+remain(?:ing)?\b",
                            "No polyps remain (polyp removal is not relevant)",
                            text,
                            count=1,
                        )
                        tl = text.lower()
            elif ra == "no":
                if "not all polyps have been removed" not in tl and "no complete removal" not in tl:
                    new_t, n_sub = re.subn(
                        r"\b(?:polyps?\s+remain\s+(?:present|in\s+place|in\s+the\s+colon)|residual\s+polyps?\s+(?:remain(?:\s+present)?|noted|present)|some\s+polyps?\s+remain|no\s+polyps?\s+remain|polyp\s+removal\s+is\s+not\s+(?:applicable|relevant))\b",
                        "not all polyps have been removed (polyp removal: no)",
                        text,
                        flags=re.I,
                        count=1,
                    )
                    if n_sub > 0:
                        text = new_t
                        tl = text.lower()
                    else:
                        additions.append("not all polyps have been removed (polyp removal: no)")
            elif ra == "yes":
                if "all polyps have been removed" not in tl and "completely removed" not in tl:
                    additions.append("all polyps have been removed")

    if "text_presence" in det_cats:
        t_atoms = sorted(set(canon_pred.get("text_presence", [])))
        if t_atoms:
            ta = t_atoms[0]
            has_neg = bool(
                re.search(
                    r"\bno\s+(?:visible\s+)?text\b|\btext\s+is\s+not\b|\bwithout\s+(?:any\s+)?text\b|\bno\s+text,\b|\bno\s+text\s+or\b",
                    tl,
                )
            )
            has_pos = ("text" in tl) and not has_neg
            if ta == "yes" and not has_pos:
                new_t, n_sub = re.subn(
                    r"\b[Nn]o\s+(?:visible\s+)?text\s+is\s+(?:visible|present|observed|seen|noted)",
                    "Text is visible",
                    text,
                )
                if n_sub == 0:
                    new_t, n_sub = re.subn(
                        r"\b[Nn]o\s+text\s+(?:observed|present|visible|seen|identified)",
                        "Text is visible",
                        text,
                    )
                if n_sub > 0:
                    text = new_t
                    tl = text.lower()
                else:
                    additions.append("text is visible")
            elif ta == "no" and not has_neg:
                new_t, n_sub = re.subn(
                    r"\b[Tt]ext\s+is\s+(?:visible|present|observed|seen|noted)",
                    "No text is visible",
                    text,
                )
                if n_sub > 0:
                    text = new_t
                    tl = text.lower()
                else:
                    additions.append("no text is visible")

    if "box_artifact_presence" in det_cats:
        b_atoms = sorted(set(canon_pred.get("box_artifact_presence", [])))
        if b_atoms:
            ba = b_atoms[0]
            has_neg = bool(
                re.search(r"\bno\s+(?:evidence\s+of\s+)?(?:green|black|box|artefact|artifact)", tl)
            )
            has_pos = bool(re.search(r"\b(?:box|artefact|artifact)\b", tl)) and not has_neg
            if ba == "yes" and not has_pos:
                new_t, n_sub = re.subn(
                    r"\b[Nn]o\s+(?:evidence\s+of\s+)?(?:green/black|green\s+or\s+black)\s+box(?:-like)?\s+(?:artefact|artifact)s?\s*(?:is\s+present|are\s+present|observed|detected|seen|noted)?",
                    "A green/black box artefact is present",
                    text,
                )
                if n_sub > 0:
                    text = new_t
                    tl = text.lower()
                else:
                    additions.append("a green/black box artefact is present")
            elif ba == "no" and not has_neg:
                new_t, n_sub = re.subn(
                    r"\b[Aa]\s+(?:green/black|green\s+or\s+black)\s+box(?:-like)?\s+(?:artefact|artifact)\s+is\s+present",
                    "No green/black box artefact is present",
                    text,
                )
                if n_sub > 0:
                    text = new_t
                    tl = text.lower()
                else:
                    additions.append("no green/black box artefact is present")

    if "procedure_type" in det_cats:
        p_atoms = sorted(set(canon_pred.get("procedure_type", [])))
        if p_atoms:
            pa = p_atoms[0]
            other = "gastroscopy" if pa == "colonoscopy" else "colonoscopy"
            if other in tl and pa not in tl:
                text = re.sub(rf"\b{other}\b", pa, text, flags=re.I)
                tl = text.lower()
            elif pa not in tl:
                additions.append(f"procedure is {pa}")

    if "abnormality_presence" in det_cats:
        ab_atoms = sorted(set(canon_pred.get("abnormality_presence", [])))
        missing_ab = [
            at
            for at in ab_atoms
            if at != "none"
            and (at not in tl or re.search(rf"\bno\s+(?:evidence\s+of\s+|visible\s+)?{re.escape(at)}", tl))
        ]
        if missing_ab:
            additions.append("evidence of " + ", ".join(missing_ab))

    if "polyp_count" in det_cats:
        pc_atoms = sorted(set(canon_pred.get("polyp_count", [])))
        if pc_atoms:
            pca = pc_atoms[0]
            pat_map = {
                "0": r"\bno\s+polyp|\bzero\s+polyp|\b0\s+polyp",
                "1": r"\b(?:one|1|single|a)\s+(?:[a-z0-9-]+\s+)?polyp",
                "2": r"\b(?:two|2)\s+polyp",
                "3": r"\b(?:three|3)\s+polyp",
                "4": r"\b(?:four|4)\s+polyp",
            }
            ok = bool(re.search(pat_map.get(pca, ""), tl)) and not (
                pca != "0" and re.search(r"\bno\s+polyp", tl)
            )
            if not ok:
                num_w = {
                    "0": "no polyps",
                    "1": "one polyp",
                    "2": "two polyps",
                    "3": "three polyps",
                    "4": "four polyps",
                }.get(pca, f"{pca} polyps")
                additions.append(f"{num_w} present")

    if "instrument_count" in det_cats:
        ic_atoms = sorted(set(canon_pred.get("instrument_count", [])))
        if ic_atoms:
            ica = ic_atoms[0]
            pat_map = {
                "0": r"\bno\s+(?:surgical\s+|visible\s+|foreign\s+bodies\s+or\s+)?instrument|\bzero\s+instrument|\b0\s+instrument",
                "1": r"\b(?:one|1|single|a)\s+(?:surgical\s+|visible\s+)?(?:instrument|tube|forceps|snare|clip)",
                "2": r"\b(?:two|2)\s+(?:surgical\s+)?instrument",
                "3": r"\b(?:three|3)\s+(?:surgical\s+)?instrument",
            }
            ok = bool(re.search(pat_map.get(ica, ""), tl)) and not (
                ica != "0" and re.search(r"\bno\s+(?:surgical\s+|visible\s+)?instrument", tl)
            )
            if not ok:
                num_w = {
                    "0": "no instruments",
                    "1": "one instrument",
                    "2": "two instruments",
                    "3": "three instruments",
                }.get(ica, f"{ica} instruments")
                additions.append(f"{num_w} visible")

    if additions:
        text = text.rstrip(" .") + ", " + ", ".join(additions) + "."
    return text

