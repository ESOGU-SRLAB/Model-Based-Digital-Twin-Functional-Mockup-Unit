# Dosya Adı: modelIK_solver.py (ORYANTASYON ODAKLI NİHAİ VERSİYON)

import numpy as np
from modelFKdeneme_verified import compute_fk_ur10e, matrix_to_rpy

def numeric_ik_solver(T_target, params, initial_guess=None, max_iterations=150, tolerance=1e-6):
    """
    Bu en gelişmiş versiyon, oryantasyon hatasını karar mekanizmasının
    merkezine koyar ve bütüncül bir hata metriği üzerinden adaptif adımlar atar.
    """
    q = np.zeros(6) if initial_guess is None else np.array(initial_guess)
    
    target_pos = T_target[:3, 3]
    target_rot = T_target[:3, :3]
    
    # Ağırlıklandırma faktörleri (bu sefer oryantasyon daha önemli)
    w_pos = 1.0  # Pozisyon hatası için ağırlık
    w_ori = 1.2  # Oryantasyon hatası için ağırlık (artırıldı)

    lambda_damping = 1e-3

    for iteration in range(max_iterations):
        fk_result = compute_fk_ur10e(q, params)
        current_pos = fk_result['T'][:3, 3]
        current_rot = fk_result['T'][:3, :3]
        
        # Hataları hesapla
        pos_error = target_pos - current_pos
        R_error = target_rot @ current_rot.T
        trace_R = np.clip(np.trace(R_error), -1.0, 3.0)
        angle_error = np.arccos((trace_R - 1) / 2)
        
        orient_error = np.zeros(3)
        if angle_error > 1e-6:
            axis = 1 / (2 * np.sin(angle_error)) * np.array([
                R_error[2, 1] - R_error[1, 2], R_error[0, 2] - R_error[2, 0], R_error[1, 0] - R_error[0, 1]
            ])
            orient_error = axis * angle_error
            
        # Başarı durumunu ham (ağırlıksız) hatalara göre kontrol et
        pos_error_norm = np.linalg.norm(pos_error)
        orient_error_norm = np.linalg.norm(orient_error)
        if pos_error_norm < tolerance and orient_error_norm < tolerance:
            return {'success': True, 'joint_angles': q, 'position_error': pos_error_norm,
                    'orientation_error': orient_error_norm, 'iterations': iteration + 1}

        # Ağırlıklı hata vektörünü oluştur
        error_vector = np.concatenate([w_pos * pos_error, w_ori * orient_error])
        
        # Jakobyan'ı hesapla ve ağırlıklandır
        J = compute_numerical_jacobian(q, params)
        J_weighted = np.vstack([w_pos * J[:3, :], w_ori * J[3:, :]])

        # DLS ile bir sonraki adımı hesapla
        A = J_weighted.T @ J_weighted + lambda_damping * np.eye(6)
        g = J_weighted.T @ error_vector
        delta_q = np.linalg.solve(A, g)
        
        q_new = q + delta_q
        
        # --- ADAPTİF ADIM KONTROLÜ (GELİŞTİRİLDİ) ---
        # Yeni adayın BÜTÜNCÜL hatasını hesapla
        fk_new_result = compute_fk_ur10e(q_new, params)
        new_pos_error = target_pos - fk_new_result['T'][:3, 3]
        
        R_error_new = target_rot @ fk_new_result['T'][:3, :3].T
        trace_R_new = np.clip(np.trace(R_error_new), -1.0, 3.0)
        angle_error_new = np.arccos((trace_R_new - 1) / 2)
        new_orient_axis = 1 / (2 * np.sin(angle_error_new)) * np.array([
            R_error_new[2, 1] - R_error_new[1, 2], R_error_new[0, 2] - R_error_new[2, 0], R_error_new[1, 0] - R_error_new[0, 1]
        ])
        new_orient_error = new_orient_axis * angle_error_new
        
        # Mevcut ve yeni adayın ağırlıklı hata normlarını karşılaştır
        current_total_error = np.linalg.norm(error_vector)
        new_total_error = np.linalg.norm(np.concatenate([w_pos * new_pos_error, w_ori * new_orient_error]))

        if new_total_error < current_total_error: # Toplam hata azaldıysa, adım geçerli
            q = np.arctan2(np.sin(q_new), np.cos(q_new))
            lambda_damping = max(1e-6, lambda_damping * 0.8) # Güven artar
        else: # Toplam hata arttıysa, adım geçersiz
            lambda_damping = min(1e6, lambda_damping * 2) # Panik, sönümlemeyi artır
            
        if lambda_damping >= 1e6: break

    return {'success': False, 'joint_angles': q, 'position_error': pos_error_norm,
            'orientation_error': orient_error_norm, 'iterations': max_iterations}


def compute_numerical_jacobian(q, params, epsilon=1e-6):
    J = np.zeros((6, 6))
    fk_ref = compute_fk_ur10e(q, params)
    current_rot = fk_ref['T'][:3, :3]
    for i in range(6):
        q_plus = q.copy(); q_plus[i] += epsilon
        fk_plus = compute_fk_ur10e(q_plus, params)
        q_minus = q.copy(); q_minus[i] -= epsilon
        fk_minus = compute_fk_ur10e(q_minus, params)
        J[:3, i] = (fk_plus['T'][:3, 3] - fk_minus['T'][:3, 3]) / (2 * epsilon)
        R_dot = (fk_plus['T'][:3, :3] - fk_minus['T'][:3, :3]) / (2 * epsilon)
        omega_skew = current_rot.T @ R_dot
        J[3:, i] = [(omega_skew[2, 1] - omega_skew[1, 2]) / 2, (omega_skew[0, 2] - omega_skew[2, 0]) / 2, (omega_skew[1, 0] - omega_skew[0, 1]) / 2]
    return J