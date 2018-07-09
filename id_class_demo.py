# UTF-8
'''
Author YangYang
 Anaconda python3.6
 Code by Pycharm

基于facenet的人脸识别，人脸数据库在face_human
调用工具箱id_class
*** WARNING !!!***
human_face里至少要有一类存在！不然报错！
输入的时候注意斜杠！是/！
暂时不支持中文显示、中文类名！不要用中文！

./test/7.jpg
./test/1.jpg
'''
from id_class import face_identification

id = face_identification()  # 申明Object
# 算嵌入值
test_pic = input('图片地址，注意斜杠/:').strip()  # 输入图片，忽视空格
embedding, pic, bounding_boxes, face_exist = id.cal_emb(test_pic=test_pic)  # pic: array, 原图

# 算欧氏距离
class_dict = id.cal_dist(emb=embedding)

# 查字典，对比并更新
text = id.check_datesets(pic, bounding_boxes, embedding=embedding)

id.show_faces(image=pic, boxes=bounding_boxes, text_list=text)
id.clear_all()
