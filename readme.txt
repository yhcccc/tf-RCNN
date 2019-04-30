# ��������
��������ǻ���tensorflow��tflearn��ʵ�ֲ���RCNN���ܡ�

# ��������
windows10 + python3.6 + tensorflow1.2 + tflearn + cv2 + scikit-learn + matplotlib

# ���ݼ�
����17flowers�ݼ�, �������أ�http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

# ����˵��
1��setup.py---��ʼ��·��
2��config.py---����
3��tools.py---����������ʾ����ͼ�񹤾�
4��train_alexnet.py---�����ݼ�Ԥѵ��Alexnet����
5��preprocessing_RCNN.py---ͼ��Ĵ���ѡ�������������ݴ�ȡ�ȣ�
6��selectivesearch.py---ѡ��������Դ��
7��fine_tune_RCNN.py---С���ݼ�΢��Alexnet
8��RCNN_output.py---ѵ��SVM������RCNN�����Ե�ʱ�����ͼƬѡ���7��16����û�в���ѵ���ģ�����Ļ�Ч���ã���Ϊѵ���õĶ��ǵ���ģ�

# �ļ�˵��
1��train_list.txt---Ԥѵ�����ݣ�������17flowers�ļ�����
2��fine_tune_list.txt---΢������2flowers�ļ�����
3��1.png---ֱ����ѡ�������������򻮷�
4��2.png---ͨ��RCNN������򻮷�

# ��������
1���������ݼ�С��ԭ����΢��ʱ��û��������һ����һ��bitch32����������128�����������룬�о����������٣�
2����û�ж��������ô�������ֵģ����зǼ���ֵ���Ƽ���canny����û�н��У�������
# 3����ѡ���������ֱ�ӽ������ŵģ�
4���������ݼ������Ĳ��ò�һ��������΢����ѵ��SVMʱ���õ�IOU��ֵһ�����д����Ρ�
5��û��Bounding Box Regression

# �Ķ�
1�� ��ѡ����������ż�����padding
2�� ��AlexNet��ԭ�ɺ�ԭ����һ����
3�� Ч���ܲ���

# �ο�
1�����ģ�Rich feature hierarchies for accurate object detection and semantic segmentation��https://www.computer.org/csdl/proceedings/cvpr/2014/5118/00/5118a580-abs.html��
2�����Ͳο���http://blog.csdn.net/u011534057/article/details/51218218��http://blog.csdn.net/u011534057/article/details/51218250
3������ο���http://www.cnblogs.com/edwardbi/p/5647522.html��https://github.com/edwardbi/DeepLearningModels/tree/master/RCNN

