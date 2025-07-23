# Dosya Adı: realistic_ik_performance_test.py
# Bu kod, IK çözücüsünün gerçekçi koşullarda performansını test eder.
# Hedef: 1e-3 pozisyon hatası ve %90 başarı oranı

import numpy as np
import matplotlib.pyplot as plt
from modelFKdeneme_verified import compute_fk_ur10e
from modelIK_solver import numeric_ik_solver
import time

class IKPerformanceTester:
    def __init__(self, params):
        self.params = params
        self.test_results = []
        
    def generate_reachable_targets(self, num_targets=100):
        """
        FK kullanarak ulaşılabilir rastgele hedefler oluştur
        """
        targets = []
        target_angles = []
        
        print(f"🎯 {num_targets} adet ulaşılabilir hedef oluşturuluyor...")
        
        for i in range(num_targets):
            # Rastgele geçerli eklem açıları (UR10e çalışma alanı içinde)
            random_joints = np.array([
                np.random.uniform(-2*np.pi, 2*np.pi),    # Taban dönüş
                np.random.uniform(-np.pi, 0),           # Omuz (yukarı)
                np.random.uniform(-np.pi, np.pi),       # Dirsek
                np.random.uniform(-np.pi, np.pi),       # Bilek 1
                np.random.uniform(-np.pi, np.pi),       # Bilek 2
                np.random.uniform(-2*np.pi, 2*np.pi)    # Bilek 3
            ])
            
            # FK ile hedef pozisyonu hesapla
            fk_result = compute_fk_ur10e(random_joints, self.params)
            targets.append(fk_result['T'])
            target_angles.append(random_joints)
            
        print(f"✅ {len(targets)} hedef başarıyla oluşturuldu.")
        return targets, target_angles
    
    def generate_smart_initial_guesses(self, target_position, num_guesses=5):
        """
        Hedef pozisyona göre akıllı başlangıç tahminleri oluştur
        """
        guesses = []
        
        # 1. Home pozisyondan başla
        guesses.append(np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0]))
        
        # 2. Hedefin yaklaşık yönüne göre taban açısı ayarla
        target_x, target_y = target_position[0], target_position[1]
        base_angle_estimate = np.arctan2(target_y, target_x)
        
        # Farklı konfigürasyonlar dene
        configs = [
            [base_angle_estimate, -np.pi/3, np.pi/3, 0, np.pi/2, 0],      # Elbow up
            [base_angle_estimate, -2*np.pi/3, 2*np.pi/3, 0, np.pi/2, 0],  # Elbow down
            [base_angle_estimate, -np.pi/4, np.pi/2, np.pi/4, np.pi/2, 0], # Mid config
        ]
        
        for config in configs:
            if len(guesses) < num_guesses:
                guesses.append(np.array(config))
        
        # Kalan slotları rastgele doldur
        while len(guesses) < num_guesses:
            random_guess = np.random.uniform(-np.pi, np.pi, 6)
            guesses.append(random_guess)
            
        return guesses
    
    def add_noise_to_guess(self, base_guess, noise_level_degrees=10):
        """
        Başlangıç tahminini biraz boz
        """
        noise_rad = np.radians(noise_level_degrees)
        noise = np.random.normal(0, noise_rad, 6)
        return base_guess + noise
    
    def test_single_target(self, T_target, ground_truth_angles=None, 
                          max_attempts=10, noise_level=15):
        """
        Tek bir hedef için IK çözücüyü test et
        """
        target_pos = T_target[:3, 3]
        
        # Akıllı başlangıç tahminleri oluştur
        smart_guesses = self.generate_smart_initial_guesses(target_pos)
        
        attempts = []
        best_result = None
        best_error = float('inf')
        
        start_time = time.time()
        
        # Önce akıllı tahminleri dene
        for i, base_guess in enumerate(smart_guesses[:min(max_attempts, len(smart_guesses))]):
            # Tahmine gürültü ekle
            noisy_guess = self.add_noise_to_guess(base_guess, noise_level)
            
            # IK çözücüyü çağır
            result = numeric_ik_solver(T_target, self.params, 
                                     initial_guess=noisy_guess,
                                     tolerance=1e-6, max_iterations=100)
            
            # Gerçek hatayı hesapla
            if result['success']:
                fk_check = compute_fk_ur10e(result['joint_angles'], self.params)
                pos_error = np.linalg.norm(T_target[:3, 3] - fk_check['position'])
                result['actual_position_error'] = pos_error
                
                if pos_error < best_error:
                    best_error = pos_error
                    best_result = result.copy()
                    best_result['attempt_number'] = i + 1
            
            attempts.append({
                'attempt': i + 1,
                'initial_guess': noisy_guess.copy(),
                'success': result['success'],
                'iterations': result.get('iterations', 0),
                'position_error': result.get('actual_position_error', float('inf'))
            })
            
            # İyi sonuç bulduysak erken dur
            if result['success'] and result.get('actual_position_error', float('inf')) < 1e-4:
                break
        
        solve_time = time.time() - start_time
        
        return {
            'success': best_result is not None,
            'best_result': best_result,
            'all_attempts': attempts,
            'solve_time': solve_time,
            'target_position': target_pos,
            'ground_truth_angles': ground_truth_angles
        }
    
    def run_performance_test(self, num_targets=50, max_attempts_per_target=8):
        """
        Kapsamlı performans testi
        """
        print("="*80)
        print("🚀 IK ÇÖZÜCÜ PERFORMANS TESTİ BAŞLATIYOR")
        print("="*80)
        print(f"📊 Test Parametreleri:")
        print(f"   • Hedef sayısı: {num_targets}")
        print(f"   • Hedef başına maksimum deneme: {max_attempts_per_target}")
        print(f"   • Başarı kriteri: Pozisyon hatası < 1e-3 m")
        print(f"   • Gürültü seviyesi: 15 derece")
        print()
        
        # Hedefleri oluştur
        targets, ground_truth_angles = self.generate_reachable_targets(num_targets)
        
        # Test sonuçlarını sakla
        results = {
            'successful_targets': 0,
            'failed_targets': 0,
            'position_errors': [],
            'solve_times': [],
            'iteration_counts': [],
            'attempt_counts': [],
            'detailed_results': []
        }
        
        print("🔄 Test başlıyor...")
        print()
        
        for i, (target, gt_angles) in enumerate(zip(targets, ground_truth_angles)):
            print(f"Test {i+1:3d}/{num_targets} - Hedef: {np.round(target[:3, 3], 4)}", end="")
            
            test_result = self.test_single_target(
                target, gt_angles, max_attempts_per_target
            )
            
            if test_result['success'] and test_result['best_result']['actual_position_error'] < 1e-3:
                results['successful_targets'] += 1
                results['position_errors'].append(test_result['best_result']['actual_position_error'])
                results['solve_times'].append(test_result['solve_time'])
                results['iteration_counts'].append(test_result['best_result']['iterations'])
                results['attempt_counts'].append(test_result['best_result']['attempt_number'])
                print(f" ✅ Başarılı - Hata: {test_result['best_result']['actual_position_error']*1000:.2f} mm")
            else:
                results['failed_targets'] += 1
                print(" ❌ Başarısız")
                
            results['detailed_results'].append(test_result)
        
        # Sonuçları analiz et ve raporla
        self.generate_performance_report(results, num_targets)
        return results
    
    def generate_performance_report(self, results, total_targets):
        """
        Detaylı performans raporu oluştur
        """
        success_rate = (results['successful_targets'] / total_targets) * 100
        
        print()
        print("="*80)
        print("📈 PERFORMANS RAPORU")
        print("="*80)
        
        # Ana metrikler
        print(f"🎯 BAŞARI ORANI: {success_rate:.1f}% ({results['successful_targets']}/{total_targets})")
        
        if results['position_errors']:
            pos_errors_mm = np.array(results['position_errors']) * 1000
            print(f"📏 POZİSYON HATASI İSTATİSTİKLERİ:")
            print(f"   • Ortalama: {np.mean(pos_errors_mm):.3f} mm")
            print(f"   • Medyan:   {np.median(pos_errors_mm):.3f} mm")
            print(f"   • Maksimum: {np.max(pos_errors_mm):.3f} mm")
            print(f"   • Std. Dev: {np.std(pos_errors_mm):.3f} mm")
            
            print(f"⚡ PERFORMANS METRİKLERİ:")
            print(f"   • Ortalama çözüm süresi: {np.mean(results['solve_times']):.3f} saniye")
            print(f"   • Ortalama iterasyon: {np.mean(results['iteration_counts']):.1f}")
            print(f"   • Ortalama deneme sayısı: {np.mean(results['attempt_counts']):.1f}")
        
        # Değerlendirme
        print(f"\n🔍 DEĞERLENDİRME:")
        if success_rate >= 90:
            print("   🟢 Mükemmel! Hedeflenen %90 başarı oranı aşıldı.")
        elif success_rate >= 80:
            print("   🟡 İyi! Başarı oranı kabul edilebilir seviyede.")
        else:
            print("   🔴 Geliştime gerekli! Başarı oranı hedefin altında.")
            
        if results['position_errors'] and np.max(results['position_errors']) < 1e-3:
            print("   🟢 Pozisyon hassasiyeti hedefi karşılandı (< 1mm).")
        else:
            print("   🔴 Pozisyon hassasiyeti geliştirilmeli.")
        
        print("="*80)


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Robot parametreleri
    ur10e_params = {
        'a2': 0.613,
        'a3': 0.572,
        'd4': 0.174,
        'd5': 0.120,
        'LB': 0.181,
        'LTP': 0.117
    }
    
    # Test objesi oluştur
    tester = IKPerformanceTester(ur10e_params)
    
    # Ana performans testini çalıştır
    results = tester.run_performance_test(
        num_targets=100,           # 100 rastgele hedef
        max_attempts_per_target=10  # Hedef başına max 10 deneme
    )
    
    # İsteğe bağlı: Başarısız olanları detaylandır
    failed_details = [r for r in results['detailed_results'] if not r['success']]
    if failed_details and len(failed_details) < 10:  # Sadece az başarısızlık varsa detaylandır
        print(f"\n🔍 BAŞARISIZ HEDEFLER ANALİZİ ({len(failed_details)} adet):")
        for i, fail in enumerate(failed_details[:5]):  # İlk 5 tanesi
            pos = fail['target_position']
            print(f"   {i+1}. Hedef pozisyon: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            best_error = min([att['position_error'] for att in fail['all_attempts']])
            print(f"      En iyi deneme hatası: {best_error*1000:.2f} mm")