import numpy as np
import pydicom as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_contour(contour_dcm_path, contour_npy_path):
    ds = pd.dcmread({contour_dcm_path})
    st_data = np.array([])
    for i in range(len(ds.ROIContourSequence[0].ContourSequence)):
        tmp = np.int16(ds.ROIContourSequence[0].ContourSequence[i].ContourData)
        tmp = tmp.reshape(-1, 3)
        st_data = np.append(st_data, tmp)
    st_data = st_data.reshape(-1, 3)
    print(st_data.shape)
    print(st_data[0:10])
    np.save({contour_npy_path}, st_data)

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

contour_file_path = '/home/uchiyama/work/image/Phantom/DICOMdata/CT_Contour'
contour_dcm_path = f'{contour_file_path}/ct_contour.dcm'
contour_npy_path = f'{contour_file_path}/ct_contour.npy'
#extract_contour(contour_dcm_path, contour_npy_path)
plot_contour(contour_npy_path)