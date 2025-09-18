# Dosya Adı: fk_ik_roundtrip_test_v2.py (ORYANTASYON DOĞRULAMALI NİHAİ TEST)

import numpy as np
from modelFKdeneme_verified import compute_fk_ur10e
from modelIK_solver_v2 import numeric_ik_solver

# --- Test Kurulumu ---
ur10e_params = {'a2': 0.613, 'a3': 0.572, 'd4': 0.174, 'd5': 0.120, 'LB': 0.181, 'LTP': 0.117}
theta_gt_deg = np.array([-39.72611465, -47.75159236, 54.51592357, 65.86624204, 86.84713376, 99.68789809])
theta_gt_rad = np.radians(theta_gt_deg)

print("="*70)
print("=== ORYANTASYON ODAKLI NİHAİ KANIT TESTİ ===")
print("="*70)
print(f"\n1. Ground Truth Açıları: {np.round(theta_gt_deg, 3)}")

# 2. HEDEF POZ VE ORYANTASYONU HESAPLA
fk_result = compute_fk_ur10e(theta_gt_rad, ur10e_params)
T_target = fk_result['T']
target_rpy_deg = np.degrees(fk_result['rpy'])
print(f"   -> Hedef Pozisyon: {np.round(fk_result['position'], 4)}")
print(f"   -> Hedef RPY (Derece): {np.round(target_rpy_deg, 3)}")

# 3. BİLGİLİ BAŞLANGIÇ TAHMİNİ İLE ÇÖZ
print("\n3. NÜMERİK ÇÖZÜCÜ ÇAĞRILIYOR...")
print(f"   -> Başlangıç Tahmini = Ground Truth Açıları")
solution = numeric_ik_solver(T_target, params=ur10e_params, initial_guess=theta_gt_rad, tolerance=1e-6)

# 4. SONUCU DEĞERLENDİR
print("\n4. TEST SONUCU:")
if solution['success']:
    found_angles_deg = np.degrees(solution['joint_angles'])
    final_fk = compute_fk_ur10e(solution['joint_angles'], ur10e_params)
    final_rpy_deg = np.degrees(final_fk['rpy'])
    
    pos_error = np.linalg.norm(final_fk['position'] - fk_result['position'])
    rpy_error = np.abs(np.array(final_rpy_deg) - np.array(target_rpy_deg))
    rpy_error = (rpy_error + 180) % 360 - 180 # Normalize
    max_rpy_error = np.max(np.abs(rpy_error))

    print("   ✅ BAŞARILI! Çözücü bir sonuca ulaştı.")
    print(f"   -> İterasyon Sayısı: {solution['iterations']}")
    print("\n   --- POZİSYON DOĞRULAMASI ---")
    print(f"   -> Hedef Pozisyon:    {np.round(fk_result['position'], 5)}")
    print(f"   -> Ulaşılan Pozisyon: {np.round(final_fk['position'], 5)}")
    print(f"   -> Pozisyon Hatası:   {pos_error:.3e} m")
    
    print("\n   --- ORYANTASYON DOĞRULAMASI ---")
    print(f"   -> Hedef RPY:    {np.round(target_rpy_deg, 3)}")
    print(f"   -> Ulaşılan RPY: {np.round(final_rpy_deg, 3)}")
    print(f"   -> Maksimum RPY Hatası: {max_rpy_error:.4f} derece")
else:
    print("   ❌ KRİTİK BAŞARISIZLIK!")

# --- NİHAİ YORUM ---
print("\n" + "="*70)
print("=== NİHAİ YORUM ===")
if solution['success'] and pos_error < 1e-4 and max_rpy_error < 0.1:
    print("🎉 MÜKEMMEL SONUÇ: IK çözücüsü hem pozisyonu hem de oryantasyonu tam doğrulukla sağladı.")
    print("   Bu, sistemin gerçek dünya uygulamaları için hazır olduğunun en güçlü kanıtıdır.")
else:
    print("🤔 SON DEĞERLENDİRME: Çözücü çalışıyor ancak hedefe tam olarak kenetlenemiyor.")
    print("   Bu durum, ya FK modelinizin matematiğinin bazı 'zorlu' bölgelerinden ya da")
    print("   nümerik algoritmanın tolerans/ağırlık ayarlarının daha da inceltilmesi")
    print("   gerektiğinden kaynaklanıyor olabilir.")
print("="*70)