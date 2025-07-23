import numpy as np
import time
from modelFKdeneme_verified import compute_fk_ur10e

class ImprovedIKPerformanceTester:
    def __init__(self, params):
        self.params = params
        self.test_results = []
        
    def generate_realistic_targets(self, num_targets=100):
        """
        UR10e'nin gerçek çalışma alanı içinde hedefler oluştur
        """
        targets = []
        target_angles = []
        
        print(f"🎯 {num_targets} adet gerçekçi hedef oluşturuluyor...")
        
        for i in range(num_targets):
            # Daha gerçekçi eklem sınırları (UR10e manuel referansı)
            random_joints = np.array([
                np.random.uniform(-np.pi, np.pi),           # Base: ±180°
                np.random.uniform(-np.pi, 0),               # Shoulder: -180° to 0°
                np.random.uniform(-np.pi, np.pi),           # Elbow: ±180°
                np.random.uniform(-np.pi, np.pi),           # Wrist1: ±180°
                np.random.uniform(-np.pi, np.pi),           # Wrist2: ±180°
                np.random.uniform(-np.pi, np.pi)            # Wrist3: ±180°
            ])
            
            # FK ile hedef hesapla
            fk_result = compute_fk_ur10e(random_joints, self.params)
            
            # Workspace sınırlarını kontrol et
            pos = fk_result['T'][:3, 3]
            reach = np.linalg.norm(pos[:2])  # XY düzlemindeki uzaklık
            
            # UR10e çalışma alanı: yaklaşık 1.3m menzil
            if 0.2 < reach < 1.2 and 0.0 < pos[2] < 1.5:  # Makul Z yüksekliği
                targets.append(fk_result['T'])
                target_angles.append(random_joints)
                
            if len(targets) >= num_targets:
                break
        
        print(f"✅ {len(targets)} geçerli hedef oluşturuldu.")
        return targets, target_angles
    
    def test_single_target(self, T_target, ground_truth_angles=None, max_attempts=5):
        """
        Tek hedef için gelişmiş test
        """
        from modelIK_solver_v4 import multi_start_improved_ik_solver
        
        target_pos = T_target[:3, 3]
        start_time = time.time()
        
        # Çoklu başlangıç ile çöz
        solutions = multi_start_improved_ik_solver(
            T_target, self.params, num_starts=max_attempts,
            max_iterations=200, tolerance=1e-6
        )
        
        solve_time = time.time() - start_time
        
        # En iyi çözümü seç
        valid_solutions = [sol for sol in solutions if sol['success']]
        
        if valid_solutions:
            # En düşük pozisyon hatalı olanı seç
            best_solution = min(valid_solutions, key=lambda x: x['position_error'])
            return {
                'success': True,
                'best_result': best_solution,
                'solve_time': solve_time,
                'target_position': target_pos,
                'num_solutions': len(valid_solutions)
            }
        else:
            return {
                'success': False,
                'best_result': solutions[0] if solutions else None,
                'solve_time': solve_time,
                'target_position': target_pos,
                'num_solutions': 0
            }
    
    def run_comprehensive_test(self, num_targets=50, max_attempts_per_target=5):
        """
        Kapsamlı performans testi
        """
        print("="*80)
        print("🚀 İYİLEŞTİRİLMİŞ IK ÇÖZÜCÜ PERFORMANS TESTİ")
        print("="*80)
        print(f"📊 Test Parametreleri:")
        print(f"   • Hedef sayısı: {num_targets}")
        print(f"   • Hedef başına maksimum başlangıç: {max_attempts_per_target}")
        print(f"   • Başarı kriteri: Pozisyon hatası < 1e-6 m")
        print(f"   • Maksimum iterasyon: 200")
        print()
        
        # Gerçekçi hedefler oluştur
        targets, ground_truths = self.generate_realistic_targets(num_targets)
        
        # Sonuç saklayıcıları
        results = {
            'successful': 0,
            'failed': 0,
            'position_errors': [],
            'solve_times': [],
            'iterations': [],
            'detailed_results': []
        }
        
        print("🔄 Test başlıyor...")
        print()
        
        for i, (target, gt_angles) in enumerate(zip(targets, ground_truths)):
            print(f"Test {i+1:3d}/{num_targets} - Hedef: {np.round(target[:3, 3], 4)}", end="")
            
            test_result = self.test_single_target(target, gt_angles, max_attempts_per_target)
            
            if test_result['success']:
                results['successful'] += 1
                best = test_result['best_result']
                results['position_errors'].append(best['position_error'])
                results['solve_times'].append(test_result['solve_time'])
                results['iterations'].append(best['iterations'])
                
                print(f" ✅ Başarılı - Hata: {best['position_error']*1000:.3f} mm, "
                      f"İter: {best['iterations']}, Süre: {test_result['solve_time']:.3f}s")
            else:
                results['failed'] += 1
                print(f" ❌ Başarısız - Süre: {test_result['solve_time']:.3f}s")
            
            results['detailed_results'].append(test_result)
        
        # Rapor oluştur
        self.generate_comprehensive_report(results, num_targets)
        return results
    
    def generate_comprehensive_report(self, results, total_targets):
        """
        Detaylı performans raporu
        """
        success_rate = (results['successful'] / total_targets) * 100
        
        print()
        print("="*80)
        print("📈 KAPSAMLI PERFORMANS RAPORU")
        print("="*80)
        
        print(f"🎯 BAŞARI ORANI: {success_rate:.1f}% ({results['successful']}/{total_targets})")
        
        if results['position_errors']:
            errors_mm = np.array(results['position_errors']) * 1000
            times = np.array(results['solve_times'])
            iterations = np.array(results['iterations'])
            
            print(f"\n📏 POZİSYON DOĞRULUĞU:")
            print(f"   • Ortalama hata: {np.mean(errors_mm):.4f} mm")
            print(f"   • Medyan hata:   {np.median(errors_mm):.4f} mm")
            print(f"   • Maksimum hata: {np.max(errors_mm):.4f} mm")
            print(f"   • %95 hata:      {np.percentile(errors_mm, 95):.4f} mm")
            
            print(f"\n⚡ PERFORMANS METRİKLERİ:")
            print(f"   • Ortalama süre:      {np.mean(times):.3f} ± {np.std(times):.3f} saniye")
            print(f"   • Medyan süre:        {np.median(times):.3f} saniye")
            print(f"   • Ortalama iterasyon: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
            print(f"   • Medyan iterasyon:   {np.median(iterations):.1f}")
            
            print(f"\n📊 DAĞILIM ANALİZİ:")
            print(f"   • Hata < 0.001 mm: {np.sum(errors_mm < 0.001)} adet")
            print(f"   • Hata < 0.01 mm:  {np.sum(errors_mm < 0.01)} adet")
            print(f"   • Hata < 0.1 mm:   {np.sum(errors_mm < 0.1)} adet")
        
        print(f"\n🔍 GENEL DEĞERLENDİRME:")
        if success_rate >= 90:
            print("   🟢 Mükemmel performans! Endüstriyel kullanım için hazır.")
        elif success_rate >= 80:
            print("   🟡 İyi performans. Küçük iyileştirmelerle hazır olabilir.")
        elif success_rate >= 60:
            print("   🟠 Orta performans. Algoritma geliştirme gerekli.")
        else:
            print("   🔴 Yetersiz performans. Ciddi revizyon gerekli.")
        
        if results['position_errors']:
            max_error_mm = np.max(errors_mm)
            if max_error_mm < 0.01:
                print("   🟢 Pozisyon hassasiyeti mükemmel (< 0.01mm).")
            elif max_error_mm < 0.1:
                print("   🟡 Pozisyon hassasiyeti iyi (< 0.1mm).")
            else:
                print("   🔴 Pozisyon hassasiyeti yetersiz.")
        
        print("="*80)


# Test çalıştırma scripti
if __name__ == "__main__":
    ur10e_params = {
        'a2': 0.613,
        'a3': 0.572,
        'd4': 0.174,
        'd5': 0.120,
        'LB': 0.181,
        'LTP': 0.117
    }
    
    tester = ImprovedIKPerformanceTester(ur10e_params)
    results = tester.run_comprehensive_test(
        num_targets=50,
        max_attempts_per_target=5
    )