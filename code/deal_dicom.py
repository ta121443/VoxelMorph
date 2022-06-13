from asyncore import read
import numpy as np
import pydicom as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def extract_contour(contour_dcm_path, contour_npy_path):
    ds = pd.dcmread(contour_dcm_path)
    st_data = np.array([])
    for i in range(len(ds.ROIContourSequence[0].ContourSequence)):
        tmp = np.int16(ds.ROIContourSequence[0].ContourSequence[i].ContourData)
        tmp = tmp.reshape(-1, 3)
        st_data = np.append(st_data, tmp)
    st_data = st_data.reshape(-1, 3)
    np.save(contour_npy_path, st_data)

def plot_contour(contour_npy_path):
    contour = np.load(contour_npy_path)
    X = contour[:,0]
    Y = contour[:,1]
    Z = contour[:,2]
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot(X,Y,Z, marker='o', linestyle='None')
    plt.show()

def readDicom2jpeg(dcm_fnm, save_fnm):
    ds = pd.dcmread(dcm_fnm)
    wc = ds.WindowCenter
    ww = ds.WindowWidth
    img = ds.pixel_array

    max = wc + ww/2
    min = wc - ww/2

    #ウィンドウ処理
    img = 255 * (img - min)/(max - min)
    img[img > 255] = 255
    img[img < 0] = 0

    #img = img[220:300,255:335]
    cv2.imwrite(save_fnm, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

if __name__ == '__main__':

    """dcm_path = '/home/uchiyama/work/VoxelMorph/MR-MR-Person3/contour.dcm'
    vxm_path = '/home/uchiyama/work/VoxelMorph'
    plot_contour(contour_npy_path)
    j=0
    for i in range(36, 58):
        readDicom2jpeg(f'{dcm_path}/CT/CT_{i+1}.dcm', f'{vxm_path}/CT_{j}.jpg')
        j+=1"""
    
    contour_file_path = '/home/uchiyama/work/VoxelMorph/MR-MR/Person1'
    contour_dcm_path = f'{contour_file_path}/contour.dcm'
    contour_npy_path = f'{contour_file_path}/contour.npy'
    extract_contour(contour_dcm_path, contour_npy_path)
    plot_contour(contour_npy_path)
    