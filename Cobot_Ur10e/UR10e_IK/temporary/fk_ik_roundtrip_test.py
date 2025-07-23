# Dosya Adı: fk_ik_roundtrip_test.py (ANALİTİK ÇÖZÜCÜ İÇİN GÜNCELLENDİ)

import numpy as np
# Güvenilir FK modelini ve YENİ ANALİTİK IK çözücüsünü import et
from modelFKdeneme_verified import compute_fk_ur10e
from modelIK_analytic_solver import analytic_ik_solver

# --- Test Kurulumu ---
ur10e_params = {
    'a2': 0.613, 'a3': 0.572, 'd4': 0.174, 'd5': 0.120, 'LB': 0.181, 'LTP': 0.117
}

# Test edilecek başlangıç açıları (DERECE)
theta_gt_deg = np.array([-39.726, -47.752, 54.516, 65.866, 86.847, 99.688])
theta_gt_rad = np.radians(theta_gt_deg)

# --- Test Başlangıcı ---
print("="*70)
print("=== ANALİTİK IK İLE KİNEMATİK ROUND-TRIP TESTİ ===")
print("="*70)
print(f"\n1. BAŞLANGIÇ AÇILARI (Ground Truth)")
print(f"   -> Derece: {np.round(theta_gt_deg, 3)}")
print(f"   -> Radyan: {np.round(theta_gt_rad, 4)}")

# --- ADIM A: İLERİ KİNEMATİK (AÇI -> POZ) ---
print("\n2. İLERİ KİNEMATİK HESAPLAMASI")
fk_result = compute_fk_ur10e(theta_gt_rad, ur10e_params)
T_target = fk_result['T']
print(f"   -> Hedef Pozisyon [x,y,z]: {np.round(fk_result['position'], 4)}")
print(f"   -> Hedef RPY [r,p,y]:   {np.round(fk_result['rpy'], 4)}")

# --- ADIM B: ANALİTİK TERS KİNEMATİK (POZ -> AÇI) ---
print("\n3. ANALİTİK TERS KİNEMATİK HESAPLAMASI")
print("   -> 8 farklı teorik çözüm aranıyor...")

solutions = analytic_ik_solver(T_target, ur10e_params)

print(f"   -> Toplam {len(solutions)} adet potansiyel çözüm bulundu.")

if not solutions:
    print("   ❌ Hiçbir çözüm bulunamadı!")
else:
    best_solution = None
    min_diff = float('inf')

    print("\n   === BULUNAN ÇÖZÜMLERİN DOĞRULAMASI ===")
    for i, sol in enumerate(solutions):
        found_angles_rad = sol['joint_angles']
        found_angles_deg = np.degrees(found_angles_rad)
        
        # Açı farkını, periyodikliği göz önünde bulundurarak hesapla [-180, 180]
        angle_diff = found_angles_deg - theta_gt_deg
        angle_diff = (angle_diff + 180) % 360 - 180
        
        # Ortalama farkı, en iyi çözümü bulmak için bir metrik olarak kullan
        mean_abs_diff = np.mean(np.abs(angle_diff))

        print(f"   Çözüm {i+1}:")
        print(f"     -> Bulunan Açılar (Derece): {np.round(found_angles_deg, 2)}")
        print(f"     -> Başlangıca olan fark:    {np.round(angle_diff, 2)}")
        
        if mean_abs_diff < min_diff:
            min_diff = mean_abs_diff
            best_solution = sol

    # En iyi çözümü analiz et
    best_angles_deg = np.degrees(best_solution['joint_angles'])
    best_angle_diff = (best_angles_deg - theta_gt_deg + 180) % 360 - 180
    max_single_joint_error = np.max(np.abs(best_angle_diff))

    print(f"\n   === EN İYİ EŞLEŞEN ÇÖZÜM ===")
    print(f"   -> En iyi çözümün açıları: {np.round(best_angles_deg, 3)}")
    print(f"   -> Başlangıca olan farkı:   {np.round(best_angle_diff, 3)}")
    
    # Başarı kriteri: Tek bir eklemdeki maksimum fark 1 dereceden az ise başarılı say.
    if max_single_joint_error < 1.0:
        print(f"   ✅ ROUND-TRIP BAŞARILI! (Maksimum tek eklem hatası: {max_single_joint_error:.4f}°)")
    else:
        print(f"   ⚠️  ROUND-TRIP BAŞARISIZ! (Maksimum tek eklem hatası: {max_single_joint_error:.4f}°)")

# --- ADIM C: FK DOĞRULAMA ---
if solutions and best_solution:
    print(f"\n4. İLERİ KİNEMATİK İLE EN İYİ ÇÖZÜMÜN DOĞRULANMASI")
    
    # En iyi çözümü FK ile doğrula
    verification_fk = compute_fk_ur10e(best_solution['joint_angles'], ur10e_params)
    
    pos_error = np.linalg.norm(verification_fk['position'] - fk_result['position'])
    
    print(f"   -> Orijinal Hedef Pozisyon: {np.round(fk_result['position'], 5)}")
    print(f"   -> Doğrulama Pozisyonu:     {np.round(verification_fk['position'], 5)}")
    print(f"   -> Pozisyon Hatası: {pos_error:.2e} m")
    
    if pos_error < 1e-5:
        print(f"   ✅ FK DOĞRULAMA BAŞARILI! Pozisyonlar eşleşiyor.")
    else:
        print(f"   ⚠️  FK DOĞRULAMA BAŞARISIZ! Pozisyonlar arasında fark var.")

# --- ÖZET ---
print("\n" + "="*70)
print("=== TEST ÖZET ===")
print("="*70)
if solutions and best_solution and max_single_joint_error < 1.0 and pos_error < 1e-5:
    print("🎉 SONUÇ: ANALİTİK IK ÇÖZÜCÜ BAŞARIYLA ÇALIŞIYOR!")
    print(f"   • Başlangıç konfigürasyonuna en yakın çözüm bulundu (maks hata: {max_single_joint_error:.4f}°).")
    print(f"   • Toplam {len(solutions)} farklı geçerli çözüm bulundu.")
    print("   • FK-IK round-trip doğrulaması geçti.")
else:
    print("❌ SONUÇ: ANALİTİK IK ÇÖZÜCÜDE VEYA MODELDE UYUMSUZLUK VAR!")
    if not solutions:
        print("   • Hiçbir çözüm üretilemedi.")
    else:
        print("   • Üretilen çözümler başlangıç konfigürasyonu ile eşleşmiyor.")
        print("   • Veya bulunan çözümün FK'sı orijinal hedefi sağlamıyor.")
    print("   • Kinematik model parametreleri veya formüller kontrol edilmeli.")

print("="*70)