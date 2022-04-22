import sys
import numpy as np
import os, sys
import tensorflow as tf
assert tf.__version__.startswith('2.')
import voxelmorph as vxm
import neurite as ne
import matplotlib.pyplot as plt
import cv2

def vxm_data_generator(x_data, batch_size=32):
    vol_shape = x_data.shape[1:]
    ndims = len(vol_shape)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        outputs = [fixed_images, zero_phi]

        yield(inputs, outputs)

def plot_hisotry(hist, data_path, nb_epochs, loss_name='loss'):
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('loss')
    plt.legend()
    plt.savefig(f'{data_path}/loss_{img_size}_{nb_epochs}.png')
    plt.show()

def main(img_size):
    #データのインポート
    x_train = np.load(f'/home/uchiyama/work/image/ellipse/{img_size}/before.npy')
    x_train = x_train.squeeze()

    #データの分割
    nb_val = 1000
    nb_test = 100
    x_val = x_train[-nb_val:, ...]
    x_train = x_train[:-nb_val, ...]
    x_test = x_train[-nb_test:, ...]
    x_train = x_train[:-nb_test, ...]
    #print(f'train:{x_train.shape}, validation:{x_val.shape}, test:{x_test.shape}')

    """
    #サンプルの表示
    nb_vis = 5
    idx = np.random.choice(x_train.shape[0], nb_vis, replace=False)
    example_digits = [f for f in x_train[idx, ...]]
    ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True);
    """
    #数値の正規化
    x_train = x_train.astype('float') / 255
    x_val = x_val.astype('float') / 255
    x_test = x_test.astype('float') / 255

    #U-Netの設定
    ndim = 2
    unet_input_features = 2
    inshape = (*x_train.shape[1:], unet_input_features)
    nb_features = [
        [32, 32, 32, 32],
        [32, 32, 32, 32, 32, 16]
    ]

    unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)
    print('input shape: ', unet.input.shape)
    print('output shape: ', unet.output.shape)

    disp_tensor = tf.keras.layers.Conv2D(ndim, kernel_size=3, padding='same', name='disp')(unet.output)
    def_model = tf.keras.Model(unet.inputs, disp_tensor)
    spatial_transformer = vxm.layers.SpatialTransformer(name = 'transformer')
    moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)
    moved_image_tensor = spatial_transformer([moving_image, disp_tensor])

    outputs = [moved_image_tensor, disp_tensor]
    vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)

    inshape = x_train.shape[1:]
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

   #lossの定義
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    lambda_param = 0.05
    loss_weights = [1, lambda_param]

    #voxelmorphモデルのコンパイル
    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

    #データを整理
    train_generator = vxm_data_generator(x_train)
    in_sample, out_sample = next(train_generator)
    images = [img[0,:,:,0] for img in in_sample + out_sample]
    titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
    #ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

    #学習の実行
    nb_epochs = 50
    steps_per_epoch = 100
    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=1)

    #lossをグラフ化
    data_path = f'/home/uchiyama/work/VoxelMorph/data/{img_size}/{nb_epochs}epochs'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    plot_hisotry(hist, data_path, nb_epochs)

    #lossデータをcsvファイルに保存
    losspoint = hist.history['loss']
    np.savetxt(f'{data_path}/losspoint_{img_size}_{nb_epochs}.csv', losspoint, header='losspoint')

    #検証データで試す
    val_generator = vxm_data_generator(x_val, batch_size=1)
    val_input, _ = next(val_generator)
    val_pred = vxm_model.predict(val_input)

    #valのイメージ表示
    images = [img[0,:,:,0] for img in val_input + val_pred]
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

    ne.plot.flow([val_pred[1].squeeze()], width=5)

    #検証データの保存、可視化
    moving = val_input[0].squeeze() * 255
    fixed = val_input[1].squeeze() * 255
    moving[moving < 0] = 0
    moving[moving > 255] = 255
    fixed[fixed < 0] = 0
    fixed[fixed > 255] = 255
    moved = val_pred[0].squeeze()
    moved *= 255
    moved[moved < 0] = 0
    moved[moved > 255] = 255
    cv2.imwrite(f'{data_path}/val_moving_{img_size}.png', moving)
    cv2.imwrite(f'{data_path}/val_fixed_{img_size}.png', fixed)
    cv2.imwrite(f'{data_path}/val_moved_{img_size}.png', moved)
    np.save(f'{data_path}/val_input_{img_size}', val_input)
    np.save(f'{data_path}/val_flow_{img_size}', val_pred[1].squeeze())

    #テストデータで試す
    test_generator = vxm_data_generator(x_test, batch_size=1)
    test_input, _ = next(test_generator)
    test_pred = vxm_model.predict(test_input)

    images = [img[0,:,:,0] for img in test_input + test_pred]
    titles = ['moving', 'fixed', 'moved', 'flow']
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
    ne.plot.flow([test_pred[1].squeeze()], width=5)

    #テストデータの保存、可視化
    moving = test_input[0].squeeze() * 255
    fixed = test_input[1].squeeze() * 255
    moving[moving < 0 ] = 0
    moving[moving > 255] = 255
    fixed[fixed < 0] = 0
    fixed[fixed > 255] = 255
    moved = test_pred[0].squeeze()
    moved *= 255
    moved[moved < 0] = 0
    moved[moved > 255] = 255
    flow = test_pred[1].squeeze()
    cv2.imwrite(f'{data_path}/test_moving_{img_size}.png', moving)
    cv2.imwrite(f'{data_path}/test_fixed_{img_size}.png', fixed)
    cv2.imwrite(f'{data_path}/test_moved_{img_size}.png', moved)
    np.save(f'{data_path}/test_input_{img_size}', test_input)
    np.save(f'{data_path}/test_flow_{img_size}', flow)
    
        
img_size = 32
main(img_size)