import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import copy
from PIL import Image
from torch import masked_select


Hs = {}

def blend(img1, img2):
    mask1 = (img1[:, :, 0] == 0) & (img1[:, :, 1] == 0) & (img1[:, :, 2] == 0)
    mask2 = (img2[:, :, 0] == 0) & (img2[:, :, 1] == 0) & (img2[:, :, 2] == 0)
    # plt.imshow(mask2.astype(int))
    # plt.show()
    return cv2.bitwise_or(img1, img1, mask = mask2.astype(np.uint8)) + img2




def mult(A, B):
    # A[:2, :2] = np.array([[1, 0], [0, 1]])
    # B[:2, :2] = np.array([[1, 0], [0, 1]])
    tmp1 = A[:2, :2] @ B[:2, :2]
    tmp2 = A[:2, :2] @ B[:2, 2] + A[:2, 2]
    return np.concatenate((tmp1, tmp2.reshape(2, 1)), axis=1)

# print(mult(np.array([[1, 0, 0], [0, 1, 0]]), np.array([[1, 0, 0], [0, 1, 0]])))



t = time.time()
for j in os.listdir('dump_match_pairs'):
    if(j[-3:] == 'npz'):
        data = (np.load(os.path.join('dump_match_pairs', j)))
        pts1 = []
        pts2 = []

        a = 1
        b = 1

        ctr = 0
        if(len(data['match_confidence'][data['match_confidence']>0.8])<20):
            continue

        for ind, i in enumerate(data['matches']):
            if(i != -1):
                pts1.append((a*data['keypoints0'][ind][0], b*data['keypoints0'][ind][1]))
                pts2.append((a*data['keypoints1'][i][0], b*data['keypoints1'][i][1]))
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        H, _ = cv2.estimateAffine2D(pts1, pts2, ransacReprojThreshold=5)
        H_, _ = cv2.estimateAffine2D(pts2, pts1, ransacReprojThreshold=5)
        H[0][0] = 1 if abs(H[0][0]-1)<.1 else H[0][0]
        H[1][1] = 1 if abs(H[1][1]-1)<.1 else H[1][1]
        H[0][1] = 0 if abs(H[0][1])<.1 else H[0][1]
        H[1][0] = 0 if abs(H[1][0])<.1 else H[1][0]
        H_[0][0] = 1 if abs(H_[0][0]-1)<.1 else H_[0][0]
        H_[1][1] = 1 if abs(H_[1][1]-1)<.1 else H_[1][1]
        H_[0][1] = 0 if abs(H_[0][1])<.1 else H_[0][1]
        H_[1][0] = 0 if abs(H_[1][0])<.1 else H_[1][0]
        Hs[(int(j.split('_')[0]), int(j.split('_')[1]))] = H_
        Hs[(int(j.split('_')[1]), int(j.split('_')[0]))] = H



base = 9
touched = {base}

p_H = {}

# p_H[base] = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

for i in range(100):
    touched_tmp = copy.copy(touched)
    for base1 in touched:
        for i in list(Hs.keys()):
            if(i[0] == base):
                p_H[i[1]] = Hs[i]
                touched_tmp.add(i[1])
            if(i[0]==base1 and i[1] not in touched_tmp):
                # Hs[i] = mult(Hs[i], p_H[base1])
                p_H[i[1]] = mult(Hs[i], p_H[base1])
                touched_tmp.add(i[1])
            # if(i[1] == base and i[0] not in touched_tmp):
            #     Hs[i] = np.linalg.inv(Hs[i])@p_H[base]
            #     p_H[i[0]] = Hs[i]
            #     touched_tmp.add(i[0])
    touched = touched_tmp

print(len(touched))
base_img = np.zeros((5000, 5000, 3), dtype=np.uint8)



x_disp = 1000
y_disp = 2000
H_tmp, _ = cv2.estimateAffine2D(np.float32([[0, 0], [0, 1], [1, 0], [1, 1]]), np.float32([[x_disp, y_disp], [x_disp, 1+y_disp], [1+x_disp, y_disp], [1+x_disp, 1+y_disp]]))
H_tmp[0][0] = 1 if abs(H_tmp[0][0]-1)<.01 else H_tmp[0][0]
H_tmp[1][1] = 1 if abs(H_tmp[1][1]-1)<.01 else H_tmp[1][1]
H_tmp[0][1] = 0 if abs(H_tmp[0][1])<.01 else H_tmp[0][1]
H_tmp[1][0] = 0 if abs(H_tmp[1][0])<.01 else H_tmp[1][0]

#translation
for i in p_H.keys():
    p_H[i] = mult(H_tmp, p_H[i])


base_image = cv2.resize(cv2.imread('./images/{}.jpg'.format(base), cv2.IMREAD_UNCHANGED), (640, 480))


base_img[y_disp:y_disp+base_image.shape[0], x_disp:x_disp+base_image.shape[1]] = base_image


for i in p_H.keys():
    if(i==base):
        continue
    cv2.imread('./images/{}.jpg'.format(i))
    img = cv2.resize(cv2.imread('./images/{}.jpg'.format(i), cv2.IMREAD_UNCHANGED), (640, 480))
    tmp = cv2.warpAffine(img, p_H[i], (5000, 5000))
    base_img = blend(base_img, tmp)
plt.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
# plt.savefig('./added/added_{}.png'.format(i))
plt.show()
# print(len(Hs))
# for i in Hs.keys():
#     print(i)
#     img1 = cv2.imread('./images/{}.jpg'.format(i[0]))
#     img2 = cv2.imread('./images/{}.jpg'.format(i[1]))
#     img1 = cv2.resize(img1, (640, 480))
#     img2 = cv2.resize(img2, (640, 480))
#     tmp = cv2.warpPerspective(img1, Hs[i], (1000, 1000))
#     tmp[:img2.shape[0], :img2.shape[1]] = img2
#     plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
#     plt.show()