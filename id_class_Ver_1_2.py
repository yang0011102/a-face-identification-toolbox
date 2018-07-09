# UTF-8
'''
Author YangYang
 Anaconda python3.6
 Code by Pycharm

基于facenet的人脸识别，人脸数据库在face_human

*** WARNING !!!***
human_face里至少要有一类存在！不然报错！
输入的时候注意斜杠！是/！
暂时不支持中文显示、中文类名！不要用中文！

./test/8.jpg
./test/1.jpg
'''
import os

import numpy as np
import tensorflow as tf
from lib import facenet, detect_face
from scipy import misc
import cv2


class face_identification:
    def __init__(self, gpu_memory_fraction=0.7,
                 log_device_placement=False,
                 allow_soft_placement=True,
                 # device='cpu',
                 # device_id=0,
                 cuda_gpu_set="-1",  # CUDA设置程序对多块GPU可见，-1：cpu
                 margin=44, image_size=160,
                 model_path='./model/20180408-102900/20180408-102900.pb',
                 save_or_not=True,
                 main_dir='./human_face', threshold=1):
        self.class_dict = {}  # 类名字典
        self.Text = []  # 名字
        self.place = 0  # 特征位置计数
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_gpu_set

        self.margin = margin
        self.image_size = image_size
        self.model_path = model_path
        self.save_or_Not = save_or_not
        self.main_dir = main_dir
        self.Threshold = threshold
        self.graph = tf.Graph()  # tf图
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, )
        config = tf.ConfigProto(gpu_options=gpu_options,
                                log_device_placement=log_device_placement,
                                allow_soft_placement=allow_soft_placement,
                                # device_count={device: device_id}
                                )
        self.sess = tf.Session(config=config, graph=self.graph,
                               )  # tf会话
        with self.graph.as_default():
            # Load the model
            facenet.load_model(self.model_path)
            # 获取嵌入值
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)

    # 对齐人脸
    def load_and_align_data(self, image_path):
        '''
        :param image_path: 图片的地址

        :return: images: 对齐后的人脸集合，list
                 bounding_boxes：人脸框集合 ndarray,n*5,[x1 x2 y1 y2 p], float32
                 face_exist：是否存在人脸，布尔值，True：存在；False：不存在
                 img：读取的图片
        '''
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        face_exist = True
        images = []
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_path.remove(image_path)
            print("没检测到人脸")
            face_exist = False
        else:  # 检测到人脸
            for per_box in bounding_boxes:
                box = per_box[0:4]
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(box[0] - self.margin / 2, 0)
                bb[1] = np.maximum(box[1] - self.margin / 2, 0)
                bb[2] = np.minimum(box[2] + self.margin / 2, img_size[1])
                bb[3] = np.minimum(box[3] + self.margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                images.append(prewhitened)
        return images, img, bounding_boxes, face_exist

    # 只画框
    def show_face(self, image, box):
        box = box[0:4].astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB to BGR
        cv2.imshow('ID', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 画框标字
    def show_faces(self, image, boxes, text_list):
        '''
        :param image: 图片
        :param boxes: 边界框
        :param text_list: 框上的字,list-str
        :return:

        '''
        for box, text in zip(boxes, text_list):
            box = box[0:4].astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
            cv2.putText(image, text, (box[2], box[3]), font, 0.5, (255, 255, 255), 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB to BGR
        cv2.imshow('ID', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 算嵌入值
    def cal_emb(self, test_pic):
        if test_pic.lower() == 'end':  # 跳出
            print('跳出')
        else:
            # 对齐人脸
            images, pic, bounding_boxes, face_or = self.load_and_align_data(image_path=test_pic)
            if not face_or:  # 没检测到人脸，跳出
                print('没检测到人脸')
            else:
                feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                embedding = self.sess.run(self.embeddings, feed_dict=feed_dict)  # 计算嵌入，通俗叫法是特征

        return embedding, pic, bounding_boxes, face_or

    # 算欧氏距离
    def cal_dist(self, emb):
        for embedding in emb:  # 这是特征向量，一张脸一个，array, shape:(512,)
            # 开始算欧氏距离
            for index in os.listdir(self.main_dir):
                class_dir = self.main_dir + '/' + index + '/'
                class_names = os.listdir(class_dir)  # <class 'list'>，某一类内所有文件的名称
                name_list = []
                for npy_dir in class_names:
                    if npy_dir.endswith(".npy"):  # .npy的文件名
                        name_list.append(npy_dir)
                vector_dir = class_dir + name_list[0]  # 第一个向量地址
                base_vector = np.load(vector_dir)  # 载入特征向量
                dist = np.sqrt(np.sum(np.square(np.subtract(embedding, base_vector))))  # 计算欧氏距离
                self.class_dict[index] = dist  # 动态存储到字典里
        return self.class_dict

    # 查字典，对比并更新
    def check_datesets(self, pic, bounding_boxes, embedding):
        # 如果不想要循环查找，可以用  id_class_Ver_1.00.py
        for emb in embedding:
            min_class = min(self.class_dict.items(), key=lambda x: x[1])
            if self.class_dict[min_class[0]] > self.Threshold:
                print("Error: 无匹配")

                if self.save_or_Not:
                    self.show_face(image=pic, box=bounding_boxes[self.place, :])
                    name = input('新类别的名称\n :').strip()
                    if name.lower() == 'end':
                        print("I don't want to create a new class.")  # 不创建新类别
                    else:  # 创建新类别
                        self.Text.append(name)
                        exist = os.path.exists('./human_face/' + name)
                        if not exist:
                            print('目录不存在，创建一个新的目录')
                            os.makedirs('./human_face/' + name)
                        path = "./human_face/" + name + "/"
                        print(emb.shape)
                        np.save(path + name + ".npy", emb)
            else:
                self.Text.append(min_class[0])
        return self.Text

    # 清空
    def clear_all(self):
        self.class_dict.clear()  # 清空字典
        self.Text.clear()
        self.place = 0

    def close_session(self):
        self.sess.close()  # 关闭会话
