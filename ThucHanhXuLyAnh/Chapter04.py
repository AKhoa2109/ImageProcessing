import numpy as np
import cv2
L = 256
def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Bước 1 và 2: 
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 và phần mở rộng
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin
    fp = fp/(L-1)

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4:
    # Tính DFT    
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)

    # Tính spectrum
    S = np.sqrt(F[:,:,0]**2 + F[:,:,1]**2)
    S = np.clip(S, 0, L-1)
    imgout = S.astype(np.uint8)
    return imgout


def CreateMoireFilter(M, N):
    H = np.ones((M,N), np.complex64)
    H.imag = 0.0

    u1, v1 = 44, 55
    u2, v2 = 85, 55
    u3, v3 = 41, 111
    u4, v4 = 81, 111

    u5, v5 = M-44, M-55
    u6, v6 = M-85, M-55
    u7, v7 = M-41, M-111
    u8, v8 = M-81, M-111

    D0 = 10
    for u in range(0,M):
        for v in range(0,N):
            # u1, v1
            Duv = np.sqrt((1.0*u-u1)**2 + (1.0*v-v1)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            # u2, v2
            Duv = np.sqrt((1.0*u-u2)**2 + (1.0*v-v2)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            # u3, v3
            Duv = np.sqrt((1.0*u-u3)**2 + (1.0*v-v3)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            # u4, v4
            Duv = np.sqrt((1.0*u-u4)**2 + (1.0*v-v4)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            # u5, v5
            Duv = np.sqrt((1.0*u-u5)**2 + (1.0*v-v5)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0
            
            Duv = np.sqrt((1.0*u-u6)**2 + (1.0*v-v6)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            Duv = np.sqrt((1.0*u-u7)**2 + (1.0*v-v7)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0

            Duv = np.sqrt((1.0*u-u8)**2 + (1.0*v-v8)**2)
            if Duv <= D0:
                H.real[u,v] = 0.0
    return H

def DrawInferenceFilter(imgin):
    M, N = imgin.shape
    # Bước 1
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateInferenceFilter(P, Q)
    HR = H[:,:,0]
    HR = HR*(L-1)
    imgout = HR.astype(np.uint8)
    return imgout

def CreateInferenceFilter(M,N):
    H = np.ones((M,N), np.complex64)
    H.imag = 0.0
    D0 = 7
    D1 = 7
    for u in range(0,M):
        for v in range(0,N):
            if u not in range(M//2-D0,M//2+D0+1):
                if abs(v-N//2) <= D1:
                    H.real[u,v] = 0.0
    return H

# Loại bỏ 
def RemoveMoire(imgin):
    M,N = imgin.shape
    H = CreateMoireFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout
# Loại bỏ nhiễu giao thoa
def RemoveInterference(imgin):
    M,N = imgin.shape
    H = CreateInferenceFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

# def RemoveMoireSimple(imgin):
#     M, N = imgin.shape
#     # Bước 1
#     P = cv2.getOptimalDFTSize(M)
#     Q = cv2.getOptimalDFTSize(N)
#     fp = np.zeros((P, Q), np.float32)
#     # Bước 2
#     fp[:M,:N] = 1.0*imgin

#     # Bước 3
#     for x in range(0,M):
#         for y in range(0,N):
#             if(x+y) % 2 == 1:
#                 fp[x,y] = -fp[x,y]
#     # Bước 4
#     F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
#     # Bước 5: Tạo bộ lọc H
#     H = CreateMoireFilter(P, Q)
#     # Bước 6: G = F*H
#     G = cv2.mulSpectrums(F, H, flags=cv2.DFT_ROWS)

#     # Bước 7: IDFT
#     g = cv2.idft(G,flags=cv2.DFT_SCALE)
#     # Bước 8: 
#     gR = g[:M,:N,0]
#     for x in range(0,M):
#         for y in range(0,N):
#             if(x+y) % 2 == 1:
#                 gR[x,y] = -gR[x,y]
#     gR = np.clip(gR,0,L-1)
#     imgout = gR.astype(np.uint8)
#     return imgout

def RemoveMoireSimple(imgin):
    M, N = imgin.shape
    # Bước 1: Tìm kích thước tối ưu cho DFT
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    
    # Bước 2: Đưa ảnh vào phần đầu của fp
    fp[:M, :N] = imgin.astype(np.float32)
    
    # Bước 3: Nhân (-1)^(x+y) để dịch chuyển tần số
    for x in range(M):
        for y in range(N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]
    
    # Bước 4: Tính DFT (trả về mảng 2 kênh)
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Bước 5: Tạo bộ lọc H và chuyển sang định dạng 2 kênh
    H_complex = CreateMoireFilter(P, Q)
    H = np.zeros((P, Q, 2), np.float32)  # Tạo mảng 2 kênh
    H[:, :, 0] = H_complex.real  # Kênh 0: phần thực
    H[:, :, 1] = H_complex.imag  # Kênh 1: phần ảo
    
    # Bước 6: Nhân phổ F với bộ lọc H
    G = cv2.mulSpectrums(F, H, flags=cv2.DFT_ROWS)
    
    # Bước 7: Tính IDFT
    g = cv2.idft(G, flags=cv2.DFT_SCALE)
    
    # Bước 8: Đảo ngược phép nhân (-1)^(x+y)
    gR = g[:M, :N, 0]  # Lấy phần thực
    for x in range(M):
        for y in range(N):
            if (x + y) % 2 == 1:
                gR[x, y] = -gR[x, y]
    
    # Bước 9: Chuẩn hóa và chuyển về ảnh uint8
    gR = np.clip(gR, 0, L-1)
    imgout = gR.astype(np.uint8)
    return imgout

# def RemoveInferenceFilter(imgin):
#     M, N = imgin.shape
#     # Bước 1
#     P = cv2.getOptimalDFTSize(M)
#     Q = cv2.getOptimalDFTSize(N)
#     fp = np.zeros((P, Q), np.float32)
#     # Bước 2
#     fp[:M,:N] = 1.0*imgin

#     # Bước 3
#     for x in range(0,M):
#         for y in range(0,N):
#             if(x+y) % 2 == 1:
#                 fp[x,y] = -fp[x,y]
#     # Bước 4
#     F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
#     # Bước 5: Tạo bộ lọc H
#     H = CreateInferenceFilter(P, Q)
#     # Bước 6: G = F*H
#     G = cv2.mulSpectrums(F, H, flags=cv2.DFT_ROWS)

#     # Bước 7: IDFT
#     g = cv2.idft(G,flags=cv2.DFT_SCALE)
#     # Bước 8: 
#     gR = g[:M,:N,0]
#     for x in range(0,M):
#         for y in range(0,N):
#             if(x+y) % 2 == 1:
#                 gR[x,y] = -gR[x,y]
#     gR = np.clip(gR,0,L-1)
#     imgout = gR.astype(np.uint8)
#     return imgout
def RemoveInferenceFilter(imgin):
    M, N = imgin.shape
    # Bước 1: Tìm kích thước tối ưu cho DFT
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    
    # Bước 2: Đưa ảnh vào phần đầu của fp và chuẩn hóa
    fp[:M, :N] = imgin
    
    # Bước 3: Nhân (-1)^(x+y) để dịch chuyển tần số
    for x in range(M):
        for y in range(N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]
    
    # Bước 4: Tính DFT
    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Bước 5: Tạo bộ lọc H và chuyển sang định dạng 2 kênh
    H_complex = CreateInferenceFilter(P, Q)
    H = np.zeros((P, Q, 2), np.float32)
    H[:, :, 0] = H_complex.real  # Phần thực
    H[:, :, 1] = H_complex.imag  # Phần ảo
    
    # Bước 6: Nhân phổ F với bộ lọc H
    G = cv2.mulSpectrums(F, H, flags=cv2.DFT_ROWS)
    
    # Bước 7: Tính IDFT
    g = cv2.idft(G, flags=cv2.DFT_SCALE)
    
    # Bước 8: Đảo ngược lại phép nhân (-1)^(x+y)
    gR = g[:M, :N, 0]  # Lấy phần thực của kết quả
    for x in range(M):
        for y in range(N):
            if (x + y) % 2 == 1:
                gR[x, y] = -gR[x, y]
    
    # Bước 9: Chuẩn hóa và chuyển về ảnh uint8
    gR = np.clip(gR, 0, L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def FrequencyFiltering(imgin, H):
    # Không cần mở rộng ảnh có kích thước PxQ
    f = imgin.astype(np.float32)

    # Bước 1
    F = np.fft.fft2(f)

    # Bước 2
    F = np.fft.fftshift(F)

    # Bước 3: Nhan F voi H, ta được G
    G = F * H

    # Bước 4: Shift G ra trở lại
    G = np.fft.ifftshift(G)

    # Bước 5: IDFT
    g = np.fft.ifft2(G)
    gR = np.clip(g.real, 0, L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def CreateMotionFilter(M,N):
    H = np.zeros((M,N), np.complex64)
    T = 1.0
    a = 0.1
    b = 0.1
    phi_prev = 0.0
    for u in range(0,M):
        for v in range (0,N):
            phi = np.pi*((u-M//2)*a) + ((v-N//2)*b)

            if abs(phi) < 1.0e-6:
                phi = phi_prev

            RE = T*np.sin(phi)*np.cos(phi)/phi
            IM = -T*np.sin(phi)*np.sin(phi)/phi
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def CreateDemotionFilter(M,N):
    H = np.zeros((M,N),complex)
    a = 0.1
    b = 0.1
    T = 1
    phi_prev = 0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = np.cos(phi)/T
                IM = np.sin(phi)/T
            else:
                if np.abs(np.sin(phi)) < 1.0e-6:
                    phi = phi_prev
                RE = phi/(T*np.sin(phi))*np.cos(phi)
                IM = phi/(T*np.sin(phi))*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def CreateMotion(imgin):
    M,N = imgin.shape
    H = CreateMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def CreateDemotion(imgin):
    M,N = imgin.shape
    H = CreateDemotionFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout
def CreateDemotionNoise(imgin):
    imgout = CreateDemotion(imgin)
    imgout = cv2.medianBlur(imgin, 5)
    return imgout
