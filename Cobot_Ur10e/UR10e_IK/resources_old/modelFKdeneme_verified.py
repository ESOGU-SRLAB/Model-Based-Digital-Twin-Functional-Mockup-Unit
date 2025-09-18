import numpy as np

def dh_transform_matrix(alpha, a, d, theta):
    """
    Standart Denavit-Hartenberg (DH) parametrelerinden 
    bir dönüşüm matrisi (T) oluşturur.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),              d],
        [0,              0,                           0,                          1]
    ])

def compute_fk_ur10e(theta, params):
    """
    UR10e için standart DH parametrelerini kullanarak İleri Kinematiği hesaplar.
    Bu versiyon, analitik IK çözücü ile %100 uyumludur.

    Args:
        theta (list): Eklem açıları [t1, ..., t6] radyan cinsinden.
        params (dict): Robotun sabit parametreleri.

    Returns:
        dict: 'T', 'position', 'rpy', 'quaternion' içeren sonuç sözlüğü.
    """
    # Bu versiyon, IK çözücüsüyle tutarlı olacak şekilde, her eklemin dönüşüm matrisini
    # ayrı ayrı oluşturup çarparak tam kinematiği hesaplar.
    
    # DH Parametreleri: (alpha_{i-1}, a_{i-1}, d_i, theta_i)
    T_0_1 = dh_transform_matrix(np.pi/2, 0, params['LB'], theta[0])
    T_1_2 = dh_transform_matrix(0, -params['a2'], 0, theta[1])
    T_2_3 = dh_transform_matrix(0, -params['a3'], 0, theta[2])
    T_3_4 = dh_transform_matrix(np.pi/2, 0, params['d4'], theta[3])
    T_4_5 = dh_transform_matrix(-np.pi/2, 0, params['d5'], theta[4])
    T_5_TP = dh_transform_matrix(0, 0, params['LTP'], theta[5]) # Son adım: Eklem 5'ten Takım Plakasına (Tool Plate)

    # Taban'dan Takım Plakasına (Base to Tool-Plate) olan toplam dönüşüm
    # matrislerin sırayla çarpılmasıyla elde edilir.
    T_B_TP = T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_TP

    # Sonuçları ayıkla
    pos = T_B_TP[:3, 3]
    R = T_B_TP[:3, :3]
    rpy = matrix_to_rpy(R)
    quat = rotation_matrix_to_quaternion(R)

    return {'T': T_B_TP, 'position': pos, 'rpy': rpy, 'quaternion': quat}


# --- Yardımcı Oryantasyon Dönüşüm Fonksiyonları ---

def matrix_to_rpy(R):
    """Dönüşüm matrisini ZYX RPY açılarına çevirir."""
    sy = np.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2]) # Roll
        y = np.arctan2(-R[2, 0], sy)      # Pitch
        z = np.arctan2(R[1, 0], R[0, 0]) # Yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return (x, y, z)

def rotation_matrix_to_quaternion(R):
    """Dönüşüm matrisini quaternion'a çevirir (x, y, z, w)."""
    q = np.empty((4, ))
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q[3] = 0.25 / s                # w
        q[0] = (R[2, 1] - R[1, 2]) * s # x
        q[1] = (R[0, 2] - R[2, 0]) * s # y
        q[2] = (R[1, 0] - R[0, 1]) * s # z
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
            
    # Quaternion'u (x, y, z, w) formatında döndür
    return (q[0], q[1], q[2], q[3])


# --- ÖRNEK KULLANIM VE TEST BLOĞU ---
# Bu blok, dosya doğrudan çalıştırıldığında devreye girer.
# Başka bir dosya tarafından "import" edildiğinde çalışmaz.
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

    # Örnek eklem açıları (DERECE)
    theta_in_degrees = [-39.72611465, -47.75159236, 54.51592357, 65.86624204, 86.84713376, 99.68789809]
    
    # DERECE'den RADYAN'a çevir
    theta_test_rad = np.radians(theta_in_degrees)
    
    print("--- modelFKdeneme.py Test Çalıştırması ---")
    print(f"Test Açıları (derece): {np.round(theta_in_degrees, 3)}")
    print(f"Test Açıları (radyan): {np.round(theta_test_rad, 3)}")

    # İleri Kinematik fonksiyonunu çağır (radyan değerlerle)
    fk_result = compute_fk_ur10e(theta_test_rad, ur10e_params)

    # Sonuçları yazdır
    print("\nİleri Kinematik Sonucu (Taban'dan Takım Plakasına):")
    print("Dönüşüm Matrisi (T):")
    print(np.round(fk_result['T'], 4))
    print("\nPozisyon (x,y,z):")
    print(np.round(fk_result['position'], 4))
    print("\nRPY Açıları (roll, pitch, yaw) (radyan):")
    print(np.round(fk_result['rpy'], 4))
    print("\nQuaternion (x, y, z, w):")
    print(np.round(fk_result['quaternion'], 4))