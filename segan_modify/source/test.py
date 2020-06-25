from data_loader import pre_emph, pre_emph_np, de_emph
from scipy.io import wavfile
import tensorflow as tf
import numpy as np
import time

def pre_emph_test(coeff, canvas_size):
    x_ = tf.compat.v1.placeholder(tf.float32, shape=[canvas_size,])
    x_preemph = pre_emph(x_, coeff)
    return x_, x_preemph

def pre_emph_test_np(coeff, wav):
    x_preemph = pre_emph_np(wav, coeff)
    return x_preemph
    
def clean_serving(x, canvas_size, batch_size, preemph, predictor=None):
    """ clean a utterance x
        x: numpy array containing the normalized noisy waveform
    """
    c_res = None
    print('start timer')
    Time1 = time.time()
    for beg_i in range(0, x.shape[0], canvas_size):
        if x.shape[0] - beg_i  < canvas_size:
            length = x.shape[0] - beg_i
            pad = canvas_size - length
        else:
            length = canvas_size
            pad = 0
        x_ = np.zeros((batch_size, canvas_size))
        if pad > 0:
            x_[0] = np.concatenate((x[beg_i:beg_i + length], np.zeros(pad)))
        else:
            x_[0] = x[beg_i:beg_i + length]
        print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
        canvas_w = None
        # fdict = {self.gtruth_noisy[0]:x_}
        # canvas_w = self.sess.run(self.Gs[0],
        #                         feed_dict=fdict)[0]
        test_example = {'wav_and_noisy': x_.tolist()}
        if predictor == None:
            canvas_w = x_[0]
        else:
            canvas_w = predictor.predict(test_example)
        canvas_w = canvas_w.reshape((canvas_size))
        print('canvas w shape: ', canvas_w.shape)
        if pad > 0:
            print('Removing padding of {} samples'.format(pad))
            # get rid of last padded samples
            canvas_w = canvas_w[:-pad]
        if c_res is None:
            c_res = canvas_w
        else:
            c_res = np.concatenate((c_res, canvas_w))
    # deemphasize
    c_res = de_emph(c_res, preemph)
    print('finish {}'.format(time.time()-Time1))
    return c_res

# test_wav = "test.wav"
# preemph = 0.95
# canvas_size = 2**14
# batch_size = 150

# fm, wav_data = wavfile.read(test_wav)
# wave = (2./65535.) * (wav_data.astype(np.float32) - 32767) + 1.

# x_pholder, preemph_op = pre_emph_test(preemph, wave.shape[0])

# x_pholder_np = pre_emph_test_np(preemph, wave)

# clean_serving(x=x_pholder_np, canvas_size=canvas_size, batch_size=batch_size, preemph=preemph)

# print(x_pholder)
# print(x_pholder_np)
# print(x_pholder_np.shape)
# print(x_pholder_np.dtype)

t = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 8, 1024], name='input_noise')

t_dim = tf.shape(t)

print(t_dim[0])

print(tf.shape(t_dim))

# output_shape = [t_dim[0], t_dim[1]*2, 100]

# input_shape = t.get_shape()
# in_channels = input_shape[-1]
# out_channels = output_shape[-1]
# x2d = tf.expand_dims(x, 2)
# o2d = output_shape[:2] + [1] + [output_shape[-1]]

# dilation = 2

# deconv = tf.nn.conv2d_transpose(x2d, W, output_shape=o2d,
#                                 strides=[1, dilation, 1, 1])
# w_init = init
# if w_init is None:
#     w_init = xavier_initializer(uniform=uniform)
# with tf.variable_scope(name):
#     # filter shape: [kwidth, output_channels, in_channels]
#     W = tf.get_variable('W', [kwidth, 1, out_channels, in_channels],
#                         initializer=w_init
#                         )
#     try:
#         deconv = tf.nn.conv2d_transpose(x2d, W, output_shape=o2d,
#                                         strides=[1, dilation, 1, 1])
#     except AttributeError:
#         # support for versions of TF before 0.7.0
#         # based on https://github.com/carpedm20/DCGAN-tensorflow
#         deconv = tf.nn.deconv2d(x2d, W, output_shape=o2d,
#                                 strides=[1, dilation, 1, 1])
#     if bias_init is not None:
#         b = tf.get_variable('b', [out_channels],
#                             initializer=tf.constant_initializer(0.))
#         deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
#     else:
#         deconv = tf.reshape(deconv, deconv.get_shape())
#     # reshape back to 1d
#     deconv = tf.reshape(deconv, output_shape)
#     return deconv




# updates = tf.placeholder(tf.float32, [None, 1000])
# indices = tf.placeholder(tf.int32, [None])

# var_update_rows = tf.compat.v1.scatter_update(var, [0], t[0]) 
# var_update_rows = tf.compat.v1.scatter_update(z, [0], t) 
# t = tf.reshape(t, (-1, 5))

# print(t.shape)

