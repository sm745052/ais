import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import copy


Hs = {}






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
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4)
        H_, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4)
        Hs[(int(j.split('_')[0]), int(j.split('_')[1]))] = H_
        Hs[(int(j.split('_')[1]), int(j.split('_')[0]))] = H



base = 9
touched = {base}

p_H = {}

p_H[base] = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

for i in range(100):
    touched_tmp = copy.copy(touched)
    for base1 in touched:
        
        for i in list(Hs.keys()):
            if(i[0]==base1 and i[1] not in touched_tmp):
                Hs[i] = Hs[i]@p_H[base1]
                p_H[i[1]] = Hs[i]
                touched_tmp.add(i[1])
            # if(i[1] == base and i[0] not in touched_tmp):
            #     Hs[i] = np.linalg.inv(Hs[i])@p_H[base]
            #     p_H[i[0]] = Hs[i]
            #     touched_tmp.add(i[0])
    touched = touched_tmp

print(touched)
base_img = np.zeros((3000, 3000, 3), dtype=np.uint8)



x_disp = 0
y_disp = 0
H_tmp = cv2.getPerspectiveTransform(np.float32([[0, 0], [0, 1], [1, 0], [1, 1]]), np.float32([[x_disp, y_disp], [x_disp, 1+y_disp], [1+x_disp, y_disp], [1+x_disp, 1+y_disp]]))

#translation
for i in p_H.keys():
    p_H[i] = p_H[i]@H_tmp


base_image = cv2.resize(cv2.imread('./images/{}.jpg'.format(base), cv2.IMREAD_UNCHANGED), (640, 480))

print(base_image.shape)
base_img[y_disp:y_disp+base_image.shape[0], x_disp:x_disp+base_image.shape[1]] = base_image


for i in p_H.keys():
    if(i==base):
        continue
    cv2.imread('./images/{}.jpg'.format(i))
    img = cv2.resize(cv2.imread('./images/{}.jpg'.format(i), cv2.IMREAD_UNCHANGED), (640, 480))
    tmp = cv2.warpPerspective(img, p_H[i], (3000, 3000))
    base_img[:tmp.shape[0], :tmp.shape[1]] += tmp
plt.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
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