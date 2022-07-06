import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

k1, k2 = 6, 16

t = time.time()

img = cv2.resize(cv2.imread('./images/{}.jpg'.format(k1)), (640, 480))
o_img = cv2.resize(cv2.imread('./images/{}.jpg'.format(k2)), (640, 480))


data = np.load('./dump_match_pairs/' + str(k1) + '_' + str(k2) + '_matches.npz')
pts1 = []
pts2 = []

a = 1
b = 1

for ind, i in enumerate(data['matches']):
    if(i != -1):
        pts1.append((a*data['keypoints0'][ind][0], b*data['keypoints0'][ind][1]))
        pts2.append((a*data['keypoints1'][i][0], b*data['keypoints1'][i][1]))
pts1 = np.array(pts1)
pts2 = np.array(pts2)



x_disp = 0
y_disp = 2000
# H_tmp = cv2.getPerspectiveTransform(np.float32([[0, 0], [0, 1], [1, 0], [1, 1]]), np.float32([[x_disp, y_disp], [x_disp, 1+y_disp], [1+x_disp, y_disp], [1+x_disp, 1+y_disp]]))


H, _ = cv2.estimateAffine2D(pts1, pts2 + np.float32([[x_disp, y_disp]]))
# H = H@H_tmp
tmp = cv2.warpAffine(img, H, (3000, 3000))


# pts = (cv2.perspectiveTransform(np.float32([[[0, 0]]]), H))
print(np.round(H, 2))

tmp[y_disp:y_disp+o_img.shape[0], x_disp:x_disp+o_img.shape[1]] = o_img
t = time.time() - t

# cv2.circle(tmp, (int(pts1[0][0]), int(pts1[0][1])), 5, (0, 0, 255), -1)

print(t)
plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
plt.show()