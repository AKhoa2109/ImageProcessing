import os
import cv2
import numpy as np
path = 'NhanDangTraiCay/TraiCayScratch/ThanhLong/'
path_dest = 'NhanDangTraiCay/TraiCay640x640/ThanhLong/'
lst_dir = os.listdir(path)
dem = 0
for filename in lst_dir:
    print(filename)
    fullname = path + filename
    imgin = cv2.imread(fullname, cv2.IMREAD_COLOR)
    # M: width, N: height, C: channel: 3
    M, N, C = imgin.shape
    if M < N:
        imgout = np.zeros((N, N, C), np.uint8) + 255
        imgout[:M, :N, :] = imgin
    elif M > N:
        imgout = np.zeros((M, M, C), np.uint8) + 255
        imgout[:M, :N, :] = imgin
    else:
        imgout = imgin.copy()
    imgout = cv2.resize(imgout, (640, 640))
    fullname_dest = path_dest + 'ThanhLong_%03d.jpg' % dem
    dem+=1
    cv2.imwrite(fullname_dest, imgout)