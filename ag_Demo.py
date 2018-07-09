from age_and_gender_Ver_1_2 import age_and_gender
from lib import detect_face
import tensorflow as tf
import cv2

filename = './test/Benedict Cumberbatch_001.png'
'''
Mtcnn 检测人脸
'''
minsize = 20
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor

graph = tf.Graph()  # tf图
sess = tf.Session(graph=graph)
with graph.as_default():
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
box = bounding_boxes[0:3].astype(int)
for number in range(box.shape[0]):
    cv2.rectangle(image, (box[number, 0], box[number, 1]), (box[number, 2], box[number, 3]), (0, 255, 255), 1)
    face = image[box[number, 1]:box[number, 3], box[number, 0]:box[number, 2]]
'''
进行性别/年龄检测
年龄检测有点不准，建议不要使用人脸输入，人体输入似乎更好, 也可能和性别/年龄检测模型有关
'''
ag = age_and_gender(device_id='/gpu:0',
                    GENDER_LIST=['Male', 'Female'])
best_choice, second_choice = ag.predict(mode=1,  # 0: 年龄；1：性别
                                        image_file=filename,
                                        image_bound=face,
                                        use_tf_to_read=True,  # 整张图为True，优先级高于人脸图
                                        )
font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
cv2.putText(image,
            str(best_choice[0])+' '+str(best_choice[1]),
            (box[0, 2], box[0, 3]),
            font, 0.5, (255, 255, 255), 1)
# print('Best_choice is :', best_choice)
# if second_choice != None:
#     print('Second choice is', second_choice)
sess.close()
'''
画图
'''
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB to BGR
cv2.imshow('ID', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('end')
