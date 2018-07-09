# UTF-8
'''
Author YangYang
 Anaconda python3.6
 Code by Pycharm

更新数据

*** WARNING !!!***
输入的时候注意斜杠！是/！
暂时不支持中文显示、中文类名！不要用中文！

'''
import os
import numpy as np
import tensorflow as tf
from lib import facenet, detect_face
from scipy import misc
import time
import cv2

# 对齐人脸
def load_and_align_data(image_path, image_size=160, margin=44, gpu_memory_fraction=1.0):
    '''
    :param image_path: 图片的地址
    :param image_size: 对齐后图片的大小
    :param margin: 图片界限
    :param gpu_memory_fraction: GPU设定

    :return: images: 对齐后的人脸集合，list
             bounding_boxes：人脸框集合 ndarray,n*5,[x1 x2 y1 y2 p], float32
             face_exist：是否存在人脸，布尔值，True：存在；False：不存在
             img：读取的图片
    '''
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    face_exist = True

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    images = []
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        image_path.remove(image_path)
        print("没检测到人脸")
        face_exist = False
    else:  # 检测到人脸
        for per_box in bounding_boxes:
            box = per_box[0:4]
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(box[0] - margin / 2, 0)
            bb[1] = np.maximum(box[1] - margin / 2, 0)
            bb[2] = np.minimum(box[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(box[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            images.append(prewhitened)
    return images, img, bounding_boxes, face_exist
model_path = './model/20180408-102900/20180408-102900.pb'
main_dir = './human_face'  # 图片集地址
Threshold = 1  # 人脸识别的阈值，轻易不要乱动

with tf.Graph().as_default():
    with tf.Session() as sess:
        # Load the model
        facenet.load_model(model_path)
        # 获取嵌入值
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        while True:
            class_dict = {}  # 类名字典
            Text = []
            # 输入图片，忽视空格
            test_pic = input('图片地址:').strip()
            if test_pic.lower() == 'end':  # 跳出
                print("Do break")
                break
            start_time = time.time()  # 计时开始
            # 对齐人脸
            images, _, _, face_exist = load_and_align_data(image_path=test_pic)
            if not face_exist:  # 没检测到人脸，跳出
                print("Error: Can't find a human face. "
                      "Try another picture or change the parameters of MTCNN")
                continue
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            embedding = sess.run(embeddings, feed_dict=feed_dict)  # 计算嵌入，通俗叫法是特征
            name = input('Name of the new class\n :').strip()
            exist = os.path.exists('./human_face/' + name)
            if not exist:
                print("Target file doesn't exist. "
                      "Try to create a new one named as :"+name)
                os.makedirs('./human_face/' + name)
            path = "./human_face/" + name + "/"
            print('Vector :', embedding.shape)
            np.save(path + name + ".npy", embedding)
            print('Saved in :', path + name + ".npy")

print('done')