"""
Script Ekstraksi Asli: Mengambil bobot matriks Fourier B_patch dan menghitung
vektor transformasi Fourier 64-d untuk setiap sel patch langsung dari checkpoint
eksperimen M1_GeoPE (Kvasir-VQA-x1).
"""

import zipfile
import pickle
import numpy as np
import os
from collections import OrderedDict

# 1. Path ke Checkpoint Terlatih Final
CKPT_PATH = r"g:\My Drive\DOKTORAL\KvasirVQAX1\experiments\Paper2_GeoFiLM_r32\20261030-212438_M1_GeoPE_r32_s42\checkpoints\final\geofilm_head.pt"

print("="*75)
print("EKSTRAKSI BOBOT FOURIER DARI CHECKPOINT ASLI PROYEK GeoPE")
print("Path:", CKPT_PATH)
print("="*75)

z = zipfile.ZipFile(CKPT_PATH)

# Baca matriks frekuensi B_patch (Storage '0', shape (2, 32))
b_bytes = z.read('geofilm_head/data/0')
B_patch = np.frombuffer(b_bytes, dtype=np.float32).reshape(2, 32)

# Baca pe_scale (Storage '1', scalar)
pe_bytes = z.read('geofilm_head/data/1')
pe_scale = float(np.frombuffer(pe_bytes, dtype=np.float32)[0])
effective_lambda = float(np.tanh(pe_scale))

print(f"\n1. MATRIKS BASIS FOURIER TERLATIH (B_patch in R^(2 x 32)):")
print(f"   Shape: {B_patch.shape} (32 Frekuensi Terpelajar untuk Koordinat X dan Y)")
print("   5 Frekuensi Pertama Sumbu X:", np.round(B_patch[0, :5], 4))
print("   5 Frekuensi Pertama Sumbu Y:", np.round(B_patch[1, :5], 4))
print(f"\n2. BOBOT SKALA SUNTIKAN TERLATIH (pe_scale):")
print(f"   pe_scale          = {pe_scale:.6f}")
print(f"   tanh(pe_scale)    = {effective_lambda:.6f}  (~{effective_lambda*100:.1f}% kontribusi koordinat ke token visual)")

# 3. Bentuk Grid 24x24 Koordinat Kontinu (Sesuai persis dengan geofilm_bridge_core.py)
N = 24
lin_y = np.linspace(-1.0, 1.0, N)
lin_x = np.linspace(-1.0, 1.0, N)
gy, gx = np.meshgrid(lin_y, lin_x, indexing="ij")
coords_2d = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)  # (576, 2)

# 4. Hitung Transformasi Fourier Real Sesuai Rumus forward_bridge():
# proj_2d = 2.0 * pi * matmul(coords, B_patch)
# fourier_2d = [sin(proj_2d), cos(proj_2d)] in R^(576 x 64)
proj_2d = 2.0 * np.pi * np.matmul(coords_2d, B_patch)  # (576, 32)
sin_proj = np.sin(proj_2d)                            # (576, 32)
cos_proj = np.cos(proj_2d)                            # (576, 32)
fourier_2d = np.concatenate([sin_proj, cos_proj], axis=-1)  # (576, 64)

print(f"\n3. HASIL TRANSFORMASI FOURIER KESELURUHAN (fourier_2d):")
print(f"   Dimensi Tensor Stiker: {fourier_2d.shape} (576 patch, 64 dimensi Fourier)")

# 5. Ekstraksi Patch Spesifik: Lesi Polip di Kuadran Kiri-Atas (Baris 2, Kolom 2)
target_r, target_c = 2, 2
patch_idx = target_r * N + target_c
target_coord = coords_2d[patch_idx]
target_fourier = fourier_2d[patch_idx]

print("\n" + "="*75)
print(f"4. CONTOH EKSTRAKSI 1 PATCH REAL: Polip di Baris {target_r}, Kolom {target_c} (Index: {patch_idx})")
print("="*75)
print(f"[*] Koordinat Kontinu Cartesian (x, y): ({target_coord[0]:.4f}, {target_coord[1]:.4f}) -> KUTUB UPPER-LEFT")
print(f"[*] Nilai Vektor Fourier Nyata (64 Dimensi):")
print("  Sinus (32 komponen pertama):")
print("  ", np.round(target_fourier[:8], 4), "... (menampilkan 8 pertama)")
print("  Kosinus (32 komponen kedua):")
print("  ", np.round(target_fourier[32:40], 4), "... (menampilkan 8 pertama)")

# 6. Verifikasi Bukti Lingkaran Satuan (sin^2 + cos^2 = 1.0) pada Data Nyata
print("\n* Pembuktian Kekekalan Energi (sin^2 + cos^2) pada 5 Dimensi Pertama:")
for k in range(5):
    s_val = target_fourier[k]
    c_val = target_fourier[32 + k]
    norm_sq = s_val**2 + c_val**2
    print(f"  Freq {k+1}: sin={s_val:+.4f}, cos={c_val:+.4f}  -->  sin^2 + cos^2 = {norm_sq:.6f} == 1.000000")

# 7. Simpan Vektor Asli ke File NumPy (.npy) agar bisa di-load kapan saja
out_dir = r"g:\My Drive\DOKTORAL\KvasirVQAX1\GeoFiLM_Paper2"
npy_path = os.path.join(out_dir, "real_extracted_geope_fourier_576x64.npy")
np.save(npy_path, fourier_2d)
print(f"\n[OK] Seluruh 576 vektor stiker Fourier real berhasil disimpan ke:\n     {npy_path}")
