import numpy as np
from numpy.random import randint
import cv2
import os

def rotate_fit(angle, h, w):
    radian = np.radians(angle)
    sine = np.abs(np.sin(radian))
    cosine = np.abs(np.cos(radian))
    tri_mat = np.array([[cosine, sine], [sine, cosine]], np.float32)
    old_size = np.array([w,h], np.float32)
    new_size = np.ravel(np.dot(tri_mat, old_size.reshape(-1,1)))

    affine = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    affine[:2,2] += (new_size - old_size) / 2.0
    affine[:2,:] *= (old_size / new_size).reshape(-1,1)
    return affine

def random_shift(shifts):
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src + shifts.reshape(1,-1).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return affine

def expand(ratio):
    src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    return affine

def get_affine_matrix(affine):
    mat = np.eye(3)
    mat[:2,:] = affine
    return mat

def affine_transform(image):
    h,w = image.shape[:2]
    shifts_param = int(h/32)
    shifts = randint(int(-shifts_param * 4),int(shifts_param),2)
    angle = randint(-30,30)
    ratio = np.random.rand() * 0.5 + 0.8

    affine_rotate = rotate_fit(angle, h, w)
    affine_rotate = get_affine_matrix(affine_rotate)
    affine_shift = random_shift(shifts)
    affine_shift = get_affine_matrix(affine_shift)
    affine_expand = expand(ratio)
    affine_expand = get_affine_matrix(affine_expand)

    affine = np.dot(affine_shift, np.dot(affine_expand, affine_rotate))
    affine = affine[:2,:]
    after_affine = cv2.warpAffine(image, affine, (w,h), cv2.INTER_LANCZOS4)
    return after_affine, affine

def img_move(flag, img_num, size, color):
    img_path = f'/home/uchiyama/work/image/ellipse'
    if flag == 0:
        img = cv2.imread(f'{img_path}/ellipse_{size}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contour = np.load(f'{img_path}/ellipse_{size}_contour.npy')
        c_num = contour.shape[-1]
        before = np.array([])
        b_contour_sum = np.array([])

        bef_path = f'{img_path}/{size}/before_{img_num}'
        befcon_path = f'{img_path}/{size}/before_contour_{img_num}'
        if not os.path.exists(bef_path):
            os.makedirs(bef_path)
        if not os.path.exists(befcon_path):
            os.makedirs(befcon_path)

        for i in range(img_num):
            image, affine = affine_transform(img)
            before = np.append(before, image)
            before_contour = np.dot(affine, contour)
            affine = np.reshape(affine,(-1))
            b_contour_sum = np.append(b_contour_sum, before_contour)

            cv2.imwrite(f'{bef_path}/No{i}.png', image)
            np.save(f'{befcon_path}/No{i}', before_contour)
    
        before = np.reshape(before,(-1,size,size,1))
        np.save(f'{img_path}/{size}/before', before)
        
        b_contour_sum = np.reshape(b_contour_sum,(-1,2,c_num))
        np.save(f'{img_path}/{size}/before_contour', b_contour_sum)
  
    else:
        after = np.array([])
        truth = np.array([])

        if color == 1:
            os.makedirs(f'{img_path}/{size}/after_{img_num}_color')
            os.makedirs(f'{img_path}/{size}/truth_{img_num}_color')
        else:
            os.makedirs(f'{img_path}/{size}/after_{img_num}')
            os.makedirs(f'{img_path}/{size}/truth_{img_num}')

        for i in range(img_num):
            img = cv2.imread(f'{img_path}/{size}/before_{img_num}/No{i}.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            image, affine = affine_transform(img)
            affine = np.reshape(affine,(-1))
            if color == 1:
                image[image < 130] = 80
                image[image >= 130] = 200
            after = np.append(after, image)
            truth = np.append(truth, affine)

            if color == 1:
                cv2.imwrite(f'{img_path}/{size}/after_{img_num}_color/No{i}.png', image)
                np.save(f'{img_path}/{size}/truth_{img_num}_color/No{i}', affine)
            
            else:
                cv2.imwrite(f'{img_path}/{size}/after_{img_num}/No{i}.png', image)
                np.save(f'{img_path}/{size}/truth_{img_num}/No{i}', affine)

        after = np.reshape(after, (-1,size,size,1))
        truth = np.reshape(truth,(-1,6))

        if color == 1:                
            np.save(f'{img_path}/{size}/after_color',after)
            np.save(f'{img_path}/{size}/truth_color',truth)
        else:
            np.save(f'{img_path}/{size}/after',after)
            np.save(f'{img_path}/{size}/truth',truth)

if __name__ == "__main__":
    flag = 0  #0でデータセット作成、1で平行移動
    img_num = 5000 #作成する画像の枚数
    size = 32 #画像サイズ
    color = 0

    img_move(flag, img_num, size, color)