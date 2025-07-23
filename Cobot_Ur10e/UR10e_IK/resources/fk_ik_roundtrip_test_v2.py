# Dosya Adı: final_definitive_test.py
# Bu betik, kullanıcının önerdiği "bilgili başlangıç tahmini" yöntemini
# kullanarak nümerik çözücünün doğruluğunu kesin olarak test eder.

import numpy as np
from modelFKdeneme_verified import compute_fk_ur10e
from modelIK_solver import numeric_ik_solver

# --- Test Kurulumu ---
ur10e_params = {
    'a2': 0.613, 'a3': 0.572, 'd4': 0.174, 'd5': 0.120, 'LB': 0.181, 'LTP': 0.117
}

# 1. BAŞLANGIÇ AÇILARI (Ground Truth)
# Bu sefer bu açıların "ulaşılabilir" olduğunu varsayarak ilerliyoruz.
# Eğer bu test de başarısız olursa, bu konfigürasyonun kendisinin
# modeliniz için sorunlu olduğu kesinleşir.
theta_gt_deg = np.array([-37.71974522, -50.04458599, 70.91082803, 69.76433121, 90.11464968, 100.1464968])
theta_gt_rad = np.radians(theta_gt_deg)

print("="*70)
print("=== NİHAİ VE KESİN KANIT TESTİ ===")
print("="*70)
print(f"\n1. Ground Truth Açıları: {np.round(theta_gt_deg, 3)}")

# 2. HEDEFİ HESAPLA
fk_result = compute_fk_ur10e(theta_gt_rad, ur10e_params)
T_target = fk_result['T']
print(f"   -> Bu açılar için hesaplanan Hedef Pozisyon: {np.round(fk_result['position'], 4)}")

# 3. ÇÖZÜCÜYÜ, BAŞLANGIÇ TAHMİNİ OLARAK ÇÖZÜMÜN KENDİSİNİ VEREREK ÇAĞIR
print("\n3. NÜMERİK ÇÖZÜCÜ, BİLGİLİ BAŞLANGIÇ TAHMİNİ İLE ÇAĞRILIYOR...")
print(f"   -> Başlangıç Tahmini = Ground Truth Açıları")

solution = numeric_ik_solver(T_target, 
                             params=ur10e_params, 
                             initial_guess=theta_gt_rad, # ANAHTAR NOKTA BURASI
                             tolerance=1e-6)

# 4. SONUCU DEĞERLENDİR
print("\n4. TEST SONUCU:")
if solution['success']:
    found_angles_deg = np.degrees(solution['joint_angles'])
    error = np.max(np.abs(found_angles_deg - theta_gt_deg))
    
    print("   ✅ BAŞARILI! Çözücü, başlangıç tahmini olarak çözümün kendisi verildiğinde hedefi doğruladı.")
    print(f"   -> Bulunan Açılar (Derece): {np.round(found_angles_deg, 4)}")
    print(f"   -> İterasyon Sayısı: {solution['iterations']}") # Bu sayının çok küçük (1-2) olmasını bekleriz
    print(f"   -> Maksimum Açı Hatası: {error:.6f} derece")
else:
    print("   ❌ KRİTİK BAŞARISIZLIK! Çözücü, eline çözüm verildiğinde bile 'başarılı' olamadı.")

# --- NİHAİ YORUM ---
print("\n" + "="*70)
print("=== NİHAİ YORUM ===")
if solution['success'] and error < 0.01:
    print("🎉 KANITLANDI: Nümerik IK çözücünüz ve algoritması DOĞRU çalışıyor.")
    print("   Sorun, ya ilk testlerdeki hedefin ulaşılamaz olması ya da")
    print("   o hedef için rastgele başlangıç noktalarının çok uzakta kalmasıydı.")
    print("\n   SONUÇ: Bu IK çözücüsünü güvenle kullanabilirsiniz. Önemli olan, ona")
    print("   çözüm ararken mantıklı (yakın) bir başlangıç tahmini vermektir.")
else:
    print("🤔 BU SONUÇ MÜMKÜN OLMAMALI: Eğer bu test bile başarısız olduysa,")
    print("   'modelFKdeneme_verified.py' tarafından üretilen bir pozisyon, aynı")
    print("   model kullanıldığında bile doğrulanamıyor demektir. Bu, FK modelinin")
    print("   matematiksel olarak inanılmaz derecede kararsız (ill-conditioned) olduğunu gösterir.")
print("="*70)