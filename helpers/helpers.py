import numpy as np
import cv2

def draw_line(line, img):
    height, width = np.shape(img)
    if line[0][1] != 0:
        m = -1 / np.tan(line[0][1])
        c = line[0][0] / np.sin(line[0][1])
        cv2.line(img, (0, int(c)), (width, int(m * width + c)), 255)
    else:
        cv2.line(img, (line[0][0], 0), (line[0][0], height), 255)
    
    return img


def merge_lines(lines, img):
    height, width = np.shape(img)
    for current in lines:
        if current[0][0] is None and current[0][1] is None:
            continue
        p1 = current[0][0]
        theta1 = current[0][1]
        pt1current = [None, None]
        pt2current = [None, None]

        if theta1 > np.pi * 45 / 180 and theta1 < np.pi * 135 / 180:
            pt1current[0] = 0
            pt1current[1] = p1 / np.sin(theta1)
            pt2current[0] = width
            pt2current[1] = -pt2current[0] / np.tan(theta1) + p1 / np.sin(theta1)
        else:
            pt1current[1] = 0
            pt1current[0] = p1 / np.cos(theta1)
            pt2current[1] = height
            pt2current[0] = -pt2current[1] * np.tan(theta1) + p1 / np.cos(theta1)

        for pos in lines:
            if pos[0].all() == current[0].all():
                continue
            if abs(pos[0][0] - current[0][0]) < 20 and abs(pos[0][1] - current[0][1]) < np.pi * 10 / 180:
                p = pos[0][0]
                theta = pos[0][1]
                pt1 = [None, None]
                pt2 = [None, None]

                if theta > np.pi * 45 / 180 and theta < np.pi * 135 / 180:
                    pt1[0] = 0
                    pt1[1] = p / np.sin(theta)
                    pt2[0] = width
                    pt2[1] = -pt2[0] / np.tan(theta) + p / np.sin(theta)
                else:
                    pt1[1] = 0
                    pt1[0] = p / np.cos(theta)
                    pt2[1] = height
                    pt2[0] = -pt2[1] * np.tan(theta) + p / np.cos(theta)

                if (pt1[0] - pt1current[0])**2 + (pt1[1] - pt1current[1])**2 < 64**2 and (pt2[0] - pt2current[0])**2 + (pt2[1] - pt2current[1])**2 < 64**2:
                    current[0][0] = (current[0][0] + pos[0][0]) / 2
                    current[0][1] = (current[0][1] + pos[0][1]) / 2
                    pos[0][0] = None
                    pos[0][1] = None

    lines = list(filter(lambda a : a[0][0] is not None and a[0][1] is not None, lines))
    return lines


def preprocess_image(img):
    rows = np.shape(img)[0]

    for i in range(rows):

        cv2.floodFill(img, None, (0, i), 0)
        cv2.floodFill(img, None, (i, 0), 0)
        cv2.floodFill(img, None, (rows-1, i), 0)
        cv2.floodFill(img, None, (i, rows-1), 0)
        cv2.floodFill(img, None, (1, i), 1)
        cv2.floodFill(img, None, (i, 1), 1)
        cv2.floodFill(img, None, (rows - 2, i), 1)
        cv2.floodFill(img, None, (i, rows - 2), 1)

    rowtop = None
    rowbottom = None
    colleft = None
    colright = None
    thresholdBottom = 50
    thresholdTop = 50
    thresholdLeft = 50
    thresholdRight = 50
    center = rows // 2

    for i in range(center, rows):
        if rowbottom is None:
            temp = img[i]
            if sum(temp) < thresholdBottom or i == rows-1:
                rowbottom = i
        if rowtop is None:
            temp = img[rows-i-1]
            if sum(temp) < thresholdTop or i == rows-1:
                rowtop = rows-i-1
        if colright is None:
            temp = img[:, i]
            if sum(temp) < thresholdRight or i == rows-1:
                colright = i
        if colleft is None:
            temp = img[:, rows-i-1]
            if sum(temp) < thresholdLeft or i == rows-1:
                colleft = rows-i-1

    newimg = np.zeros(np.shape(img))
    startatX = (rows + colleft - colright)//2
    startatY = (rows - rowbottom + rowtop)//2
    for y in range(startatY, (rows + rowbottom - rowtop)//2):
        for x in range(startatX, (rows - colleft + colright)//2):
            newimg[y, x] = img[rowtop + y - startatY, colleft + x - startatX]

    return newimg


def check_if_white(img):
    count  = 0
    for i in img:
        for j in i:
            if j > 250:
                count+= 1
        
    if count >= 5:
        return True
    else:
        return False


def normalize_data(data):
    
    return (data/9) - .5
    
def denormalize_data(data):
    
    return (data + .5) * 9