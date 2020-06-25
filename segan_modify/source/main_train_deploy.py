from __future__ import print_function
import tensorflow as tf
import numpy as np
from model import SEGAN, SEAE
import os
from tensorflow.python.client import device_lib
from scipy.io import wavfile
from data_loader import pre_emph
from data_loader import read_and_decode, de_emph


devices = device_lib.list_local_devices()

flags = tf.app.flags
flags.DEFINE_integer("seed", 111, "Random seed (Def: 111).")
flags.DEFINE_integer("epoch", 150, "Epochs to train (Def: 150).")
flags.DEFINE_integer("batch_size", 150, "Batch size (Def: 150).")
flags.DEFINE_integer("save_freq", 50, "Batch save freq (Def: 50).")
flags.DEFINE_integer("canvas_size", 2**14, "Canvas size (Def: 2^14).")
flags.DEFINE_integer("denoise_epoch", 5, "Epoch where noise in disc is "
                     "removed (Def: 5).")
flags.DEFINE_integer("l1_remove_epoch", 150, "Epoch where L1 in G is "
                     "removed (Def: 150).")
flags.DEFINE_boolean("bias_deconv", False,
                     "Flag to specify if we bias deconvs (Def: False)")
flags.DEFINE_boolean("bias_downconv", False,
                     "flag to specify if we bias downconvs (def: false)")
flags.DEFINE_boolean("bias_D_conv", False,
                     "flag to specify if we bias D_convs (def: false)")
# TODO: noise decay is under check
flags.DEFINE_float("denoise_lbound", 0.01,
                   "Min noise std to be still alive (Def: 0.001)")
flags.DEFINE_float("noise_decay", 0.7, "Decay rate of noise std (Def: 0.7)")
flags.DEFINE_float("d_label_smooth", 0.25, "Smooth factor in D (Def: 0.25)")
flags.DEFINE_float("init_noise_std", 0.5, "Init noise std (Def: 0.5)")
flags.DEFINE_float("init_l1_weight", 100., "Init L1 lambda (Def: 100)")
flags.DEFINE_integer("z_dim", 256, "Dimension of input noise to G (Def: 256).")
flags.DEFINE_integer("z_depth", 256, "Depth of input noise to G (Def: 256).")
flags.DEFINE_string("save_path", "segan_results", "Path to save out model "
                    "files. (Def: dwavegan_model"
                    ").")
flags.DEFINE_string(
    "g_nl", "leaky", "Type of nonlinearity in G: leaky or prelu. (Def: leaky).")
flags.DEFINE_string(
    "model", "gan", "Type of model to train: gan or ae. (Def: gan).")
flags.DEFINE_string("deconv_type", "deconv", "Type of deconv method: deconv or "
                                             "nn_deconv (Def: deconv).")
flags.DEFINE_string("g_type", "ae", "Type of G to use: ae or dwave. (Def: ae).")
flags.DEFINE_float("g_learning_rate", 0.0002, "G learning_rate (Def: 0.0002)")
flags.DEFINE_float("d_learning_rate", 0.0002, "D learning_rate (Def: 0.0002)")
flags.DEFINE_float("beta_1", 0.5, "Adam beta 1 (Def: 0.5)")
flags.DEFINE_float("preemph", 0.95, "Pre-emph factor (Def: 0.95)")
flags.DEFINE_string("synthesis_path", "dwavegan_samples", "Path to save output"
                                                          " generated samples."
                                                          " (Def: dwavegan_sam"
                                                          "ples).")
flags.DEFINE_string("e2e_dataset", "data/segan.tfrecords", "TFRecords"
                    " (Def: data/"
                    "segan.tfrecords.")
flags.DEFINE_string("save_clean_path", "test_clean_results",
                    "Path to save clean utts")
flags.DEFINE_string("test_wav", None, "name of test wav (it won't train)")
flags.DEFINE_string("weights", None, "Weights file")
flags.DEFINE_string("task_type", "train", "task type {tain, infer, eval, export}")
flags.DEFINE_string("dataset_dir", "/opt/ml/input/data/training/train/segan.tfrecords", "directory for training data")
flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
flags.DEFINE_string("sagemaker", 'false', "whehter to use this script for byos")
FLAGS = flags.FLAGS


def get_vars(self):
    t_vars = tf.trainable_variables()
    self.d_vars_dict = {}
    self.g_vars_dict = {}
    for var in t_vars:
        if var.name.startswith('d_'):
            self.d_vars_dict[var.name] = var
        if var.name.startswith('g_'):
            self.g_vars_dict[var.name] = var
    self.d_vars = self.d_vars_dict.values()
    self.g_vars = self.g_vars_dict.values()
    for x in self.d_vars:
        assert x not in self.g_vars
    for x in self.g_vars:
        assert x not in self.d_vars
    for x in t_vars:
        assert x in self.g_vars or x in self.d_vars, x.name
    self.all_vars = t_vars
    if self.d_clip_weights:
        print('Clipping D weights')
        self.d_clip = [v.assign(tf.clip_by_value(v, -0.05, 0.05))
                       for v in self.d_vars]
    else:
        print('Not clipping D weights')


def pre_emph_test(coeff, canvas_size):
    x_ = tf.placeholder(tf.float32, shape=[canvas_size, ])
    x_preemph = pre_emph(x_, coeff)
    return x_, x_preemph


def input_fn(dataset_dir='', num_epochs=1, canvas_size=32, preemph=0.1, batch_size=32):
    filename_queue = tf.train.string_input_producer([dataset_dir], num_epochs=num_epochs)
    get_wav, get_noisy = read_and_decode(filename_queue, canvas_size, preemph)

    # # try dataset API
    # dataset = tf.data.TFRecordDataset(dataset_dir)
    # dataset = dataset.batch(batch_size, drop_remainder=True)
    # dataset = dataset.map(map_func=read_and_decode, num_parallel_calls=2)

    # dataset = dataset.shuffle(buffer_size=1000)

    # if num_epochs > 1:
    #     dataset = dataset.repeat(num_epochs)
    
    # iterator = dataset.make_one_shot_iterator()

    # wavbatch_data, noisybatch_data = iterator.get_next(name='Train_IteratorGetNext')

    # load the data to input pipeline
    print(get_wav)
    wavbatch, \
        noisybatch = tf.train.shuffle_batch([get_wav,
                                             get_noisy],
                                            batch_size=batch_size,
                                            num_threads=2,
                                            capacity=1000 + 3 * batch_size,
                                            min_after_dequeue=1000,
                                            name='wav_and_noisy')
    print(wavbatch)

    num_examples = 0
    for record in tf.python_io.tf_record_iterator(dataset_dir):
        num_examples += 1
    print('!!!!!!!!!!!!!!!!total examples in TFRecords {}: {}'.format(dataset_dir,
                                                        num_examples))
    ################################################################
    labels = wavbatch
    # labels = wavbatch_data
    return {"wav_and_noisy": noisybatch}, labels
    # return {"input_noise": noisybatch_data}, labels


# def model_fn(features, wavbatch, mode, params):
def model_fn(features, labels, mode, params):
    print('Creating GAN model')
    print(features["wav_and_noisy"])
    wavbatch = labels
    input_var = None
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     # var = tf.compat.v1.get_variable("wav_and_noisy", [FLAGS.batch_size, FLAGS.canvas_size], tf.float32, initializer=tf.zeros_initializer())
    #     # var_update_rows = tf.compat.v1.scatter_update(var, [0], features["wav_and_noisy"]) 
    #     var_update_rows = features["wav_and_noisy"]
    #     var_update_rows.set_shape([FLAGS.batch_size, FLAGS.canvas_size])
    #     input_var = {"wav_and_noisy":var_update_rows}
    # else:
    #     input_var = features
    
    input_var = tf.reshape(features["wav_and_noisy"], shape=[-1, FLAGS.canvas_size])

    se_model = SEGAN(None, params, ['/gpu:0'], input_var, wavbatch)

    if mode == tf.estimator.ModeKeys.PREDICT:

        # var_update_rows = tf.compat.v1.scatter_update(var, [0], t[0]) 
        G, _  = se_model.generator(input_var, is_ref=False, spk=None,
                               do_prelu=False)
        # wavbatch = labels
        # fake_feature = {'input_noise',tf.placeholder(dtype=tf.float32, shape=[None,], name='input_noise')}
        # se_model = SEGAN(None, params, ['/GPU:0'], features, wavbatch)
        # wav_data = features["input_noise"]
        # wave = (2./65535.) * (wav_data.astype(np.float32) - 32767) + 1.
        # if params.preemph > 0:
        #     print('preemph test wave with {}'.format(params.preemph))
        #     # x_pholder, preemph_op = pre_emph_test(FLAGS.preemph, wave.shape[0])
        #     # wave = sess.run(preemph_op, feed_dict={x_pholder: wave})
        #     wave = pre_emph(wave, params.preemph)
        # print('test wave shape: ', wave.shape)
        # print('test wave min:{}  max:{}'.format(np.min(wave), np.max(wave)))
        # c_wave = se_model.clean(wave)
        # print('c wave min:{}  max:{}'.format(np.min(c_wave), np.max(c_wave)))
        # se_model.Gs[0](features['input_noise'],)
        predictions = {"clean_wav": G}
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs
        )
    elif mode == tf.estimator.ModeKeys.TRAIN:
        train_op = None
        train_op = tf.group(se_model.d_opt, se_model.g_opt)

        g_loss = se_model.g_losses[-1]
        d_loss = se_model.d_losses[-1]
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_loss', d_loss)

        predictions = {"clean_wav": se_model.Gs[-1]}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=g_loss+d_loss,
            train_op=train_op
        )
    else:
        print("wrong!! invalid mode: {}".format(mode))

def main(_):
    print('Parsed arguments: ', FLAGS.__flags)

    # # make save path if it is required
    # if not os.path.exists(FLAGS.save_path):
    #     os.makedirs(FLAGS.save_path)
    # if not os.path.exists(FLAGS.synthesis_path):
    #     os.makedirs(FLAGS.synthesis_path)
    # np.random.seed(FLAGS.seed)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # udevices = []
    # print(devices)
    # for device in devices:
    #     if len(devices) > 1 and ('CPU' in device.name or 'XLA' in device.name):
    #         # Use cpu only when we dont have gpus
    #         continue
    #     print('Using device: ', device.name)
    #     udevices.append(device.name)
    # # execute the session
    # with tf.Session(config=config) as sess:
    #     if FLAGS.model == 'gan':
    #         print('Creating GAN model')
    #         se_model = SEGAN(sess, FLAGS, udevices)
    #     elif FLAGS.model == 'ae':
    #         print('Creating AE model')
    #         se_model = SEAE(sess, FLAGS, udevices)
    #     else:
    #         raise ValueError(
    #             '{} model type not understood!'.format(FLAGS.model))
    #     if FLAGS.test_wav is None:
    #         se_model.train(FLAGS, udevices)
    #     else:
    #         if FLAGS.weights is None:
    #             raise ValueError('weights must be specified!')
    #         print('Loading model weights...')
    #         se_model.load(FLAGS.save_path, FLAGS.weights)
    #         fm, wav_data = wavfile.read(FLAGS.test_wav)
    #         wavname = FLAGS.test_wav.split('/')[-1]
    #         if fm != 16000:
    #             raise ValueError('16kHz required! Test file is different')
    #         wave = (2./65535.) * (wav_data.astype(np.float32) - 32767) + 1.
    #         if FLAGS.preemph > 0:
    #             print('preemph test wave with {}'.format(FLAGS.preemph))
    #             x_pholder, preemph_op = pre_emph_test(
    #                 FLAGS.preemph, wave.shape[0])
    #             wave = sess.run(preemph_op, feed_dict={x_pholder: wave})
    #         print('test wave shape: ', wave.shape)
    #         print('test wave min:{}  max:{}'.format(np.min(wave), np.max(wave)))
    #         c_wave = se_model.clean(wave)
    #         print('c wave min:{}  max:{}'.format(
    #             np.min(c_wave), np.max(c_wave)))
    #         wavfile.write(os.path.join(
    #             FLAGS.save_clean_path, wavname), 16e3, c_wave)
    #         print('Done cleaning {} and saved '
    #               'to {}'.format(FLAGS.test_wav,
    #                              os.path.join(FLAGS.save_clean_path, wavname)))

    # model_params = { "field_size": FLAGS.field_size,
    #     "feature_size": FLAGS.feature_size,
    #     "embedding_size": FLAGS.embedding_size,
    #     "learning_rate": FLAGS.learning_rate,
    #     "batch_norm_decay": FLAGS.batch_norm_decay,
    #     "l2_reg": FLAGS.l2_reg,
    #     "deep_layers": FLAGS.deep_layers,
    #     "dropout": FLAGS.dropout
    # }
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # Segan = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.save_path,
    #                                params=model_params, config=tf.estimator.RunConfig().replace(session_config=config))
    Segan = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.save_path, params=FLAGS,
                                   config=tf.estimator.RunConfig().replace(session_config=config))

    if FLAGS.task_type == 'train':
        print("!!!!!!!!!!!!!!!!!epoch is {}".format(FLAGS.epoch))
        for _ in range(FLAGS.epoch):
            Segan.train(input_fn=lambda: input_fn(
                dataset_dir=FLAGS.dataset_dir, canvas_size=FLAGS.canvas_size, preemph=FLAGS.preemph, num_epochs=1, batch_size=FLAGS.batch_size))
            print("finish one time!!!!!!")

        feature_spec = {
            "wav_and_noisy": tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.canvas_size], name="wav_and_noisy")
        }
        print("????feature {}".format(feature_spec["wav_and_noisy"]))
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            feature_spec)
        print(feature_spec)
        Segan.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == '__main__':
    tf.app.run()
