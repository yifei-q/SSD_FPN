import cv2
import matplotlib.cm as mpcm
import imutils

from dataset import dataset_common
import numpy as np


def gain_translate_table():
    label2name_table = {}
    for class_name, labels_pair in dataset_common.SPERM_LABELS.items():
        label2name_table[labels_pair[0]] = class_name
    return label2name_table

label2name_table = gain_translate_table()

def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2):
    shape = img.shape
    scale = 0.4
    text_thickness = 1
    line_type = 8
    centers=[]
    for i in range(bboxes.shape[0]):
        if classes[i] < 1:
            continue
        bbox = bboxes[i]
        color = colors_tableau[classes[i]]
        # Draw bounding boxes
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)

        cropped = img[p1[0]:p2[0],p1[1]:p2[1]]
        #print(cropped)
        Grayimg = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(Grayimg, 90, 255, cv2.THRESH_BINARY_INV)  ##sperm_11=90,sperm_12=85 , sperm_13=168 ,sperm_14=160 ,sperm_15=165
        cnts,h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cnts[1] if imutils.is_cv2() else cnts[0]
        if len(cnts)>0:
            for cnt in cnts:
                try:
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    centeroid = (int(x),int(y))
                    radius = int(radius)
                    if (radius>1):
                        #cv2.circle(img,(p1[1]+centeroid[0],p1[0]+centeroid[1]),2,(255,0,255),2)
                        b = np.array([[p1[1]+centeroid[0]], [p1[0]+centeroid[1]]])
                        centers.append(np.round(b))
                        """M = cv2.moments(cnt)  # 计算图像的矩
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        cv2.circle(img, (cx + p1[1], cy + p1[0]), 2, (255, 255, 255), 2)
                        b = np.array([[cx + p1[1]], [cy + p1[0]]])
                        centers.append(np.round(b))"""

                except ZeroDivisionError:
                    pass


        # Draw text
        #s = '%s/%.1f%%' % (label2name_table[classes[i]], scores[i]*100)
        s = '%s' % (label2name_table[classes[i]])
        # text_size is (width, height)
        #text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        #p1 = (p1[0] - text_size[1], p1[1])

        #cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        #cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)
        #text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        """p1 = (p1[0] - text_size[1], p1[1])

        cv2.rectangle(img, (p1[1] - thickness // 2, p1[0] - thickness - baseline),
                      (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), text_thickness,
                    line_type)"""

    return img,centers
