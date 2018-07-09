'''
Copyright @ Rude Carnie: Age and Gender Deep Learning with TensorFlow
        https://github.com/dpressel/rude-carnie

The original code was emmm... So I re-write some block to fit my tool_box for face_id.
Annotation language is Chinese and my bad English.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from model import select_model, get_checkpoint
from lib.utils import *
import os
import csv
import cv2


def get_image(filename, image_bound=None,  # image_bound: the face detected by Algorithms, ndarray
              use_tf_to_read=True):
    # In fact, I don't understand why this image should be resized as(256,256) and return a ndarray.shaped:(12,227,227,3) .
    standardize_image = tf.image.per_image_standardization  # whiten the image( Standardlize ?)
    # some parameters
    RESIZE_AOI = 256
    RESIZE_FINAL = 227
    if use_tf_to_read:
        '''
        Use tensorflow to read whole image and resize it. This block should return a ndarray. shape :(256,256,3)
        But I can hardly fit it to Mtcnn. Maybe the original author can deal with this problem.
        '''
        # Read the image file.
        image_data = tf.gfile.FastGFile(filename, 'rb').read()

        resizing = tf.image.decode_jpeg(image_data)  # (920, 690, 3)
        img = tf.image.resize_images(resizing, [RESIZE_AOI, RESIZE_AOI])
        # image = temp_sess.run(img)
        image = img.eval()
    else:
        '''
        This block accept a face (ndarray) detected by some face detection algorithm.
        Use cv2 to resize the face array.
        Attention: Accuracy might get really bad
        '''
        image = cv2.resize(image_bound, (256, 256))  # (256,256,3)

    '''
    In this block, do come options at image. 
    '''
    crops = []
    # print('Running multi-cropped image')

    h = image.shape[0]
    w = image.shape[1]
    hl = h - RESIZE_FINAL  # RESIZE_FINAL=227，h1=29
    wl = w - RESIZE_FINAL

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    # 三维矩阵中的数字均值变为0，方差变为1
    crops.append(standardize_image(crop))
    # 左右翻转
    crops.append(standardize_image(tf.image.flip_left_right(crop)))

    corners = [(0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl / 2), int(wl / 2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(standardize_image(cropped))
        flipped = standardize_image(tf.image.flip_left_right(cropped))
        crops.append(standardize_image(flipped))

    image_batch = tf.stack(crops)
    return image_batch


class age_and_gender:
    def __init__(self,
                 RESIZE_FINAL=227,
                 GENDER_LIST=['M', 'F'],
                 AGE_LIST=['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)'],
                 MAX_BATCH_SZ=128,
                 model_list=['./model/22801-20180628T013333Z-001', './model/inception-20180628T022246Z-001'],
                 class_list=['age', 'gender'],
                 model_type='inception',
                 device_id='/gpu:0',
                 checkpoint='checkpoint',
                 single_look=False,
                 face_detection_type='cascade',  # 'yolo_tiny', 'yolo_face', 'dlib'
                 target='',
                 requested_step='',
                 face_detection_model=''  # 人脸检测器的模型地址, 请使用yolo的模型
                 ):
        self.RESIZE_FINAL = RESIZE_FINAL
        self.GENDER_LIST = GENDER_LIST
        self.AGE_LIST = AGE_LIST
        self.MAX_BATCH_SZ = MAX_BATCH_SZ
        self.model_type = model_type

        self.model_list = model_list
        self.class_list = class_list
        self.device_id = device_id
        self.checkpoint = checkpoint
        self.single_look = single_look
        self.face_detection_type = face_detection_type
        self.target = target
        self.requested_step = requested_step
        self.face_detection_model = face_detection_model

    def one_of(self, fname, types):
        return any([fname.endswith('.' + ty) for ty in types])

    def resolve_file(self, fname):
        if os.path.exists(fname): return fname
        for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
            cand = fname + suffix
            if os.path.exists(cand):
                return cand
        return None

    # def classify_many_single_crop(self, sess, label_list, softmax_output, coder, images, image_files, writer,
    #                               image_bound=None):
    #     try:
    #         num_batches = math.ceil(len(image_files) / self.MAX_BATCH_SZ)
    #         pg = ProgressBar(num_batches)
    #         for j in range(num_batches):
    #             start_offset = j * self.MAX_BATCH_SZ
    #             end_offset = min((j + 1) * self.MAX_BATCH_SZ, len(image_files))
    #
    #             batch_image_files = image_files[start_offset:end_offset]
    #             print(start_offset, end_offset, len(batch_image_files))
    #             image_batch = make_multi_image_batch(batch_image_files, coder)
    #             batch_results = sess.run(softmax_output, feed_dict={images: image_batch.eval()})
    #             batch_sz = batch_results.shape[0]
    #             for i in range(batch_sz):
    #                 output_i = batch_results[i]
    #                 best_i = np.argmax(output_i)
    #                 best_choice = (label_list[best_i], output_i[best_i])
    #                 print('Guess @ 1 %s, prob = %.2f' % best_choice)
    #                 if writer is not None:
    #                     f = batch_image_files[i]
    #                     writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
    #             pg.update()
    #         pg.done()
    #     except Exception as e:
    #         print('ERROR: Failed to run all images')

    def classify_one_multi_crop(self, sess, label_list, softmax_output, coder, images, image_file, writer,
                                image_bound=None, use_tf_to_read=True):

        print('Running file %s' % image_file)
        image = get_image(image_file, image_bound=image_bound, use_tf_to_read=use_tf_to_read)
        # temp11 = image.eval()
        batch_results = sess.run(softmax_output, feed_dict={images: image.eval()})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]

        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        # print('Guess @ 1 %s, prob = %.2f' % best_choice)

        nlabels = len(label_list)
        if nlabels > 2:
            output[best] = 0
            second_best = np.argmax(output)
            second_choice = (label_list[second_best], output[second_best])
            # print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))
        else:
            second_choice = None

        if writer is not None:
            writer.writerow((image_file, best_choice[0], '%.2f' % best_choice[1]))

        return best_choice, second_choice

    def list_images(self, srcfile):
        with open(srcfile, 'r') as csvfile:
            delim = ',' if srcfile.endswith('.csv') else '\t'
            reader = csv.reader(csvfile, delimiter=delim)
            if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
                print('skipping header')
                _ = next(reader)

            return [row[0] for row in reader]

    def predict(self, image_file=None, mode=0,
                image_bound=None,
                use_tf_to_read=True):
        model_dir = self.model_list[mode]
        class_type = self.class_list[mode]
        files = []

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            label_list = self.AGE_LIST if class_type == 'age' else self.GENDER_LIST
            nlabels = len(label_list)

            model_fn = select_model(self.model_type)

            with tf.device(self.device_id):

                images = tf.placeholder(tf.float32, [None, self.RESIZE_FINAL, self.RESIZE_FINAL, 3])
                logits = model_fn(nlabels, images, 1, False)
                # init = tf.global_variables_initializer()

                requested_step = self.requested_step if self.requested_step else None

                checkpoint_path = '%s' % (model_dir)

                model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, self.checkpoint)

                saver = tf.train.Saver()
                saver.restore(sess, model_checkpoint_path)

                softmax_output = tf.nn.softmax(logits)

                coder = ImageCoder()

                # Support a batch mode if no face detection model
                # if len(files) == 0:
                #     if (os.path.isdir(filename)):
                #         for relpath in os.listdir(filename):
                #             abspath = os.path.join(filename, relpath)
                #             print(abspath)
                #             if os.path.isfile(abspath) and any(
                #                     [abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                #                 print(abspath)
                #                 files.append(abspath)
                #     else:
                #         files.append(filename)
                #         # If it happens to be a list file, read the list and clobber the files
                #         if any([filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
                #             files = self.list_images(filename)

                writer = None
                # output = None
                # if self.target:
                #     output = open(self.target, 'w')
                #     writer = csv.writer(output)
                #     writer.writerow(('file', 'label', 'score'))
                # image_files = list(filter(lambda x: x is not None, [self.resolve_file(f) for f in files]))
                # if self.single_look:
                #     self.classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer)
                #
                # else:
                    # for image_file in filename:
                best_choice, second_choice = self.classify_one_multi_crop(sess, label_list, softmax_output,
                                                                          coder, images, image_file, writer,
                                                                          image_bound=image_bound,
                                                                          use_tf_to_read=use_tf_to_read)

                # if output is not None:
                #     output.close()
        return best_choice, second_choice
