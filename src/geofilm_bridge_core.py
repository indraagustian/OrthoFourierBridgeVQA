"""geofilm_bridge_core.py — GeoFiLM (M4) + Canonical Closed-Loop Bridge.

Dasar bukti desain (semua dari data proyek ini, bukan asumsi):
  1. Probe (notebook 04): informasi kategori lokasi/ukuran/warna ADA secara linier di fitur
     visual beku -> masalahnya readout, bukan persepsi.
  2. U4 (GeoFiLM++): slot sebagai AWALAN KELUARAN tidak pindah ke prosa. Konsistensi prosa U4
     terhadap slot-nya sendiri = setara kontrol (prosa M4 yang tak pernah melihat slot itu).
     Judge hanya membaca prosa, jadi slot keluaran tidak terhitung.
  3. CL-Bridge (PolypSize): prediksi kepala bantu yang DISUNTIKKAN KE SISI ENCODER pindah ke
     prosa: polyp_size 70,5 -> 81,8 (judge Qwen), dan tiga kategori lokasi ikut naik walau
     kepalanya hanya untuk ukuran.
  4. 83% kekalahan M4 dari Gautam (-1,50 dari -1,81pp) ada di polyp_size + 3 kategori lokasi.

Desain: kepala M4 dipertahankan PERSIS (resep yang menghasilkan judge 84,30), lalu tiap
kategori ber-jawaban-tertutup mendapat kepala kanonik yang prediksinya disuntikkan ke token
visual dengan rumus CL-Bridge. Gerbang nol saat inisialisasi -> step 0 identik dengan M4.

Tidak ada pemilihan kategori dengan tangan: kategori ikut bila kosakata kanoniknya (atom
dengan frekuensi >= min_count di split TRAIN) mencakup >= min_cover jawabannya dan memiliki
>= 2 kemungkinan jawaban. Multi/single-label juga ditentukan dari data.
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

CORE_VERSION = "bridge-2026-11-03b"

# Karakter yang TIDAK BOLEH ada di atom kanonik. Kehadirannya berarti `original` terurai
# salah (bug versi lama: str(dict) -> "{'q': ..., 'a': ...}" lalu dipecah pada koma).
_FORBIDDEN_IN_ATOM = set("{}\"':?[]")


# =============================================================================
# 1. UTILITAS DATA
# =============================================================================
def as_list(x):
    """`original` -> list jawaban (nilai kunci "a").

    Kolom `original` Kvasir-VQA-x1 berisi JSON valid: '[{"q": "...", "a": "11-20mm"}, ...]'.
    BUG versi lama: setelah json.loads berhasil, elemen dict diubah dengan str(dict) sehingga
    menjadi repr Python "{'q': '...', 'a': '11-20mm'}" — pertanyaan & tanda kutip ikut masuk
    ke kosakata. Di sini setiap dict diambil nilai jawabannya saja.
    """
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
        "{'q': 'Is there text?', 'a': 'no'}": ["no"],          # bentuk repr lama (sisa U4)
        "plain answer": ["plain answer"],
    }
    for inp, exp in cases.items():
        got = as_list(inp)
        assert got == exp, f"as_list({inp!r}) = {got!r}, harus {exp!r}"
    return True


# =============================================================================
# 2. SPESIFIKASI KANONIK (ditemukan dari data TRAIN, tanpa daftar tangan)
# =============================================================================
def discover_canonical_specs(df, min_count=20, min_cover=0.99, verbose=True):
    """-> OrderedDict {kategori: {"atoms": [...], "multi": bool, ...}} (urut nama, deterministik).

    Aturan inklusi (SAMA untuk semua kategori):
      * vocab   = atom dengan frekuensi >= min_count
      * cover   = fraksi kemunculan atom yang masuk vocab           -> harus >= min_cover
      * |vocab| >= 2  (kategori berjawaban tunggal tidak membawa informasi)
      * multi   = ada jawaban dengan > 1 atom di data
    """
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
    """Mengubah (question_class, original) -> target datar untuk kepala kanonik.

    Representasi batch: y (B, T) multi-hot atas SEMUA atom semua kategori, m (B, C) mask
    "kategori ini ditanyakan & berlabel sah pada baris ini".
    STRICT: bila satu atom saja di luar kosakata, aspek itu di-mask (tidak ada label parsial).
    """

    def __init__(self, specs):
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
                continue                                   # strict: jangan beri label parsial
            ci = self.cat_index[c]
            if not self.multi[ci] and len(set(idx)) != 1:
                continue                                   # single-label wajib tepat satu atom
            out.append((ci, sorted(set(idx))))
            seen.add(c)
        return out

    def to_tensors(self, enc_list):
        B = len(enc_list)
        y = torch.zeros(B, self.T, dtype=torch.float32)
        m = torch.zeros(B, self.C, dtype=torch.float32)
        for b, enc in enumerate(enc_list):
            for ci, idx in enc:
                y[b, idx] = 1.0
                m[b, ci] = 1.0
        return y, m


def canonical_loss(logits, y, m, labeler):
    """Jumlah loss per-aspek dibagi ukuran batch (normalisasi per-BARIS).

    Alasan normalisasi ini: kontribusi gradien bantu ke backbone bersama ~ (rata-rata aspek per
    baris) x loss — setara orde besar dengan CL-Bridge yang terbukti stabil, alih-alih 17x lipat
    bila tiap kategori dirata-rata sendiri lalu dijumlah. Kepala per-kategori tetap belajar baik
    karena AdamW menormalkan skala gradien per-parameter.
    """
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
# 3. GeoFiLMModule — SALINAN PERSIS notebook 01 [A8] (resep M4 yang menghasilkan judge 84,30)
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

    def forward_bridge(self, feats, query_embeds=None, query_mask=None):
        if feats.dim() == 2 or feats.shape[1] <= 1:
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
        elif self.gate_mode == "scalar":
            gam_g = torch.tanh(self.scale_global.to(feats.dtype))
            bet_g = torch.tanh(self.gate_global.to(feats.dtype))
            gam_s = torch.tanh(self.scale_spatial.to(feats.dtype))
            bet_s = torch.tanh(self.gate_spatial.to(feats.dtype))
            tok0_new = (1.0 + gam_g) * cls_tok.unsqueeze(1) + bet_g * e_total.unsqueeze(1)
            spat_inject = (M * a_unsq) * e_geom.unsqueeze(1) + a_unsq * e_diff.unsqueeze(1)
            patches_new = (1.0 + gam_s) * patches + pe_w * pe_2d + bet_s * spat_inject
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
        return bridged, reg_loss


# =============================================================================
# 4. CanonicalBridge — generalisasi CL-Bridge ke banyak kategori
# =============================================================================
class CanonicalBridge(nn.Module):
    """Per kategori c (rumus CL-Bridge, PolypSize 02c [A12], dipertahankan):
         a_c      = softmax_patch( w_c . LN(patch) / sqrt(D) )          peta atensi spasial
         fg_c     = sum_p a_c[p] * patch[p]
         h_c      = LN_pool( (1-sig(mix_c)) cls + sig(mix_c) fg_c ) ++ area_c(4)
         logit_c  = fc_c(h_c)             -> CE (single) / BCE (multi) terhadap atom kanonik
         p_c      = softmax / sigmoid(logit_c)
         m_c      = LN( W_c [p_c ; area_c] )
         tok0    += tanh(g_c^glob) * ( m_c + sig(fs_c) * (fg_c - cls) )
         patch_p += tanh(g_c^spat) * (M * a_c[p]) * m_c
       Gerbang g = 0 saat init -> keluaran identik dengan masukan (dibuktikan di self_test).
       Kepala membaca fitur MENTAH (seperti CL-Bridge) dan menyuntik ke keluaran GeoFiLM.
       Kepala TIDAK melihat pertanyaan -> tak ada kebocoran label; decoder yang memilih
       token kategori mana yang relevan.
    """

    def __init__(self, d_in, specs):
        super().__init__()
        self.cats = list(specs.keys())
        self.multi = [bool(specs[c]["multi"]) for c in self.cats]
        self.n_atoms = [len(specs[c]["atoms"]) for c in self.cats]
        C = len(self.cats)
        self.C, self.d_in = C, d_in
        self.scale = 1.0 / math.sqrt(d_in)

        self.norm_in = nn.LayerNorm(d_in)
        self.attn_q = nn.Linear(d_in, C)                       # satu kolom per kategori
        self.fg_mix = nn.Parameter(torch.zeros(C))             # sigmoid(0)=0,5 seperti CL-Bridge
        self.norm_pool = nn.LayerNorm(d_in)
        self.fc = nn.ModuleList([nn.Linear(d_in + 4, n) for n in self.n_atoms])
        self.embed = nn.ModuleList([nn.Sequential(nn.Linear(n + 4, d_in), nn.LayerNorm(d_in))
                                    for n in self.n_atoms])
        self.fg_scale = nn.Parameter(torch.full((C,), 0.5))
        self.gate_global = nn.Parameter(torch.zeros(C))
        self.gate_spatial = nn.Parameter(torch.zeros(C))

    @staticmethod
    def _lin(x, lin):
        return F.linear(x, lin.weight.float(), None if lin.bias is None else lin.bias.float())

    @staticmethod
    def _ln(x, ln):
        return F.layer_norm(x, ln.normalized_shape, ln.weight.float(), ln.bias.float(), ln.eps)

    def forward(self, feats_raw, feats_tgt):
        """PULAU FP32: seluruh matematika bridge dijalankan dengan autocast DIMATIKAN.
        Alasan (ditemukan oleh self_test (f)): di bawah autocast, Linear skor atensi berjalan fp16
        dan meluap (inf -> NaN pada softmax) bila bobotnya membesar, SEBELUM sempat di-.float().
        Hasil identik secara matematis; hanya keluaran akhir yang dikembalikan ke dtype asal.
        """
        dt = feats_tgt.dtype
        with torch.autocast(device_type=feats_raw.device.type, enabled=False):
            fr = feats_raw.float()
            cls_raw = fr[:, 0, :]
            P = fr[:, 1:, :]
            B, M, D = P.shape

            s = self._lin(self._ln(P, self.norm_in), self.attn_q) * self.scale          # (B,M,C)
            a = torch.softmax(s, dim=1)                                                 # (B,M,C)
            fg = torch.einsum("bmc,bmd->bcd", a, P)                                     # (B,C,D)

            gate_p = torch.sigmoid(s / self.scale)
            ac = a.clamp(min=1e-8)
            ent = -(ac * torch.log(ac)).sum(dim=1) / max(math.log(M), 1.0)
            area = torch.stack([gate_p.mean(dim=1), gate_p.std(dim=1),
                                torch.tanh(s.max(dim=1)[0]), ent], dim=-1)               # (B,C,4)

            mix = torch.sigmoid(self.fg_mix.float()).view(1, -1, 1)
            h = self._ln((1.0 - mix) * cls_raw.unsqueeze(1) + mix * fg, self.norm_pool)  # (B,C,D)

            logits, m_vecs = [], []
            for ci in range(self.C):
                hc = torch.cat([h[:, ci, :], area[:, ci, :]], dim=-1)
                lg = self._lin(hc, self.fc[ci])
                logits.append(lg)
                p = torch.sigmoid(lg) if self.multi[ci] else torch.softmax(lg, dim=-1)
                e = self.embed[ci]
                m_vecs.append(self._ln(self._lin(torch.cat([p, area[:, ci, :]], dim=-1), e[0]), e[1]))
            m = torch.stack(m_vecs, dim=1)                                              # (B,C,D)

            # TANPA normalisasi 1/C: tiap kategori memakai rumus CL-Bridge UTUH. Bukti: CL-Bridge
            # (PolypSize) mempelajari tanh(g_glob)=-0,24 dan tanh(g_spat)=+0,30; dengan 1/C=1/16
            # tiap kategori dibatasi 0,0625 -> polyp_size tak bisa mencapai kekuatan yang terbukti
            # dibutuhkan. Risiko overflow ditangani pulau fp32 ini (uji kasus terburuk di self_test).
            g_g = torch.tanh(self.gate_global.float()).view(1, -1, 1)
            g_s = torch.tanh(self.gate_spatial.float())
            f_sc = torch.sigmoid(self.fg_scale.float()).view(1, -1, 1)

            glob = (g_g * (m + f_sc * (fg - cls_raw.unsqueeze(1)))).sum(dim=1, keepdim=True)
            inj = torch.einsum("bmc,bcd->bmd", a * (M * g_s).view(1, 1, -1), m)
            ft = feats_tgt.float()
            out = torch.cat([ft[:, 0:1, :] + glob, ft[:, 1:, :] + inj], dim=1)
        return out.to(dt), torch.cat(logits, dim=-1)


class GeoFiLMBridge(nn.Module):
    """GeoFiLM (M4, persis) -> CanonicalBridge. forward_bridge -> (bridged, reg_loss, logits).

    use_bridge=False -> kontrol replikasi M4 murni (bridge tidak dibangun, logits=None).
    """

    def __init__(self, d_in, specs, geo_kwargs, use_bridge=True):
        super().__init__()
        self.use_bridge = bool(use_bridge)
        self.geo = GeoFiLMModule(d_in=d_in, **geo_kwargs)
        if not self.use_bridge:
            self.bridge = None
            return
        # Pulihkan status RNG setelah membangun bridge: konsumsi RNG oleh bridge tidak boleh
        # menggeser urutan data (RandomSampler) maupun dropout -> init LoRA, init kepala GeoFiLM,
        # DAN urutan batch identik dengan run M4. Selisih hasil = efek bridge, bukan efek seed.
        _cpu = torch.get_rng_state()
        _cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.bridge = CanonicalBridge(d_in, specs)
        torch.set_rng_state(_cpu)
        if _cuda is not None:
            torch.cuda.set_rng_state_all(_cuda)

    @property
    def pe_scale(self):
        return self.geo.pe_scale

    def forward_bridge(self, feats, query_embeds=None, query_mask=None):
        geo_out, reg = self.geo.forward_bridge(feats, query_embeds=query_embeds, query_mask=query_mask)
        if self.bridge is None or feats.dim() == 2 or feats.shape[1] <= 1:
            return geo_out, reg, None
        bridged, logits = self.bridge(feats, geo_out)
        return bridged, reg, logits

    def gate_summary(self):
        b = self.bridge
        if b is None:
            return {}
        return {c: (round(float(torch.tanh(b.gate_global[i])), 4),
                    round(float(torch.tanh(b.gate_spatial[i])), 4)) for i, c in enumerate(b.cats)}

    def gate_mean_abs(self):
        b = self.bridge
        if b is None:
            return 0.0, 0.0
        return (round(float(torch.tanh(b.gate_global).abs().mean()), 5),
                round(float(torch.tanh(b.gate_spatial).abs().mean()), 5))


# =============================================================================
# 5. SELF-TEST (dijalankan di notebook sebelum training)
# =============================================================================
def self_test(specs, d_in=768, B=4, seed=0):
    torch.manual_seed(seed)
    selfcheck_as_list()
    assert_clean_specs(specs)
    lab = CanonLabeler(specs)
    head = GeoFiLMBridge(d_in, specs, dict(use_geope=True, use_diff=True, gate_mode="film"))
    x = torch.randn(B, 577, d_in)
    q = torch.randn(B, 9, d_in)
    qm = torch.ones(B, 9)

    # (a) identitas: pada init, keluaran bridge == keluaran GeoFiLM M4 persis
    geo_out, _ = head.geo.forward_bridge(x, q, qm)
    out, reg, lg = head.forward_bridge(x, q, qm)
    assert out.shape == x.shape and lg.shape == (B, lab.T)
    assert torch.equal(out, geo_out), "Bridge tidak identitas saat init — step 0 harus = M4"
    assert torch.allclose(geo_out, x, atol=1e-6), "GeoFiLM M4 tidak identitas saat init"

    # (b) gradien: loss bantu mengalir ke kepala; loss VQA-proxy mengalir ke gerbang
    y = torch.zeros(B, lab.T); m = torch.zeros(B, lab.C)
    for b in range(B):
        ci = b % lab.C
        s, e = lab.slices[ci]
        y[b, s] = 1.0; m[b, ci] = 1.0
    la, n = canonical_loss(lg, y, m, lab)
    assert n == B and torch.isfinite(la)
    proxy = out.pow(2).mean()
    (la + proxy).backward()
    g = head.bridge
    assert g.fc[0].weight.grad is not None and g.fc[0].weight.grad.abs().sum() > 0, "kepala tak dapat gradien"
    assert g.gate_global.grad is not None and g.gate_global.grad.abs().sum() > 0, "gerbang global tak dapat gradien"
    assert g.gate_spatial.grad is not None and g.gate_spatial.grad.abs().sum() > 0, "gerbang spasial tak dapat gradien"
    assert g.attn_q.weight.grad.abs().sum() > 0

    # (c) setelah gerbang dibuka, injeksi benar-benar mengubah token
    with torch.no_grad():
        g.gate_global.fill_(0.3); g.gate_spatial.fill_(0.3)
    out2, _, _ = head.forward_bridge(x, q, qm)
    assert not torch.allclose(out2, geo_out), "gerbang terbuka tapi token tak berubah"

    # (d) mask kosong -> loss 0 tanpa NaN
    la0, n0 = canonical_loss(lg.detach(), y, torch.zeros_like(m), lab)
    assert n0 == 0 and float(la0) == 0.0

    # (e) eksperimen terkontrol: init kepala GeoFiLM & status RNG sesudahnya = run M4 persis
    gk = dict(use_geope=True, use_diff=True, gate_mode="film")
    torch.manual_seed(1234); ref = GeoFiLMModule(d_in, **gk); r_ref = torch.rand(5)
    torch.manual_seed(1234); hb = GeoFiLMBridge(d_in, specs, gk); r_new = torch.rand(5)
    assert torch.equal(r_ref, r_new), "Bridge menggeser RNG -> urutan data beda dari M4"
    for (n1, p1), (n2, p2) in zip(ref.state_dict().items(), hb.geo.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), f"init GeoFiLM berbeda dari M4 pada {n1}"
    torch.manual_seed(1234); h0 = GeoFiLMBridge(d_in, specs, gk, use_bridge=False)
    o0, _, l0 = h0.forward_bridge(x, q, qm)
    assert l0 is None and h0.bridge is None and torch.allclose(o0, x, atol=1e-6), "use_bridge=False harus = M4"

    # (f) kasus terburuk fp16 lewat forward ASLI: semua kategori menyorot 1 patch yang sama
    #     (atensi kolaps penuh) dan semua gerbang jenuh (tanh=1). Harus tetap finite.
    br = CanonicalBridge(d_in, specs)
    xf = torch.randn(2, 577, d_in)
    u = torch.randn(d_in)
    xf[:, 1 + 100, :] = u                          # patch target
    with torch.no_grad():
        ln_u = F.layer_norm(u, (d_in,))
        br.attn_q.weight.copy_(ln_u.unsqueeze(0).repeat(br.C, 1) * 1e2)
        br.attn_q.bias.zero_()
        br.gate_global.fill_(20.0); br.gate_spatial.fill_(20.0)
    worst = None
    try:
        brh, xh = br.half(), xf.half()
        with torch.no_grad():
            outh, lgh = brh(xh, xh)
        assert torch.isfinite(outh).all() and torch.isfinite(lgh).all(), \
            "bridge meluap fp16 pada kasus terburuk"
        worst = round(float((outh.float() - xf).abs().max()), 1)
    except RuntimeError as e:                      # sebagian build CPU tak mendukung op half
        worst = f"skip ({str(e)[:50]})"
    return {"T": lab.T, "C": lab.C, "params": sum(p.numel() for p in head.parameters()),
            "bridge_params": sum(p.numel() for p in head.bridge.parameters()),
            "worst_case_fp16_inject": worst}
