import cv2
import imutils
import numpy as np

from helpers.helpers import *


class Preprocess:

    def __init__(self, kernel_size, iterations):
        '''
        Init function
        '''

        self.image = None
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)  
        self.iterations = iterations
        self.extracted_grid = None


    def read_img(self, image_path):
        '''
        Read the uploaded image using Open-CV
        '''

        npimg = np.fromstring(image_path, np.uint8)
        self.image = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)


    def threshold_and_invert(self):
        '''
        Using Open-CV's Adaptive Threshold function to converth the image into B&W
        '''

        converted_img = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)
        return converted_img


    def dialate_image(self, thresh_img):
        '''
        Dialating the image to join broken parts of the image. It also helps in sharpening the borderline of the sudoku
        '''

        dialated = cv2.dilate(thresh_img, self.kernel, iterations=self.iterations) 
        return dialated


    def flood_fill_image(self, dialated_img):
        '''
        Flood-filling the image to find the largest blob(border)
        '''

        outerbox = dialated_img
        maxi = - 1
        maxpt = None
        value = 10
        height, width = np.shape(outerbox)

        for y in range(height):
            row = dialated_img[y]
            for x in range(width):
                if row[x] >= 128:
                    area = cv2.floodFill(outerbox, None, (x, y), 64)[0]
                    if value > 0:
                        value -= 1
                    if area > maxi:
                        maxpt = (x, y)
                        maxi = area

        cv2.floodFill(outerbox, None, maxpt, (255, 255, 255))

        for y in range(height):
            row = dialated_img[y]
            for x in range(width):
                if row[x] == 64 and x != maxpt[0] and y != maxpt[1]:
                    cv2.floodFill(outerbox, None, (x, y), 0)

        return outerbox, height, width


    def erode_image(self, box_highlighed_image):
        '''
        Reverting the process of dialation
        '''

        eroded = cv2.erode(box_highlighed_image, self.kernel, iterations=self.iterations) 
        return eroded


    def draw_lines_on_image(self, eroded_image):
        '''
        Draw lines on the image
        '''

        lines = cv2.HoughLines(eroded_image, 1, np.pi / 180, 200)
        tmpimg = np.copy(eroded_image)
        for i in range(len(lines)):
            tmpimp = draw_line(lines[i], tmpimg)

        return tmpimp, lines


    def find_extreme_lines(self, lines, lined_image):
        '''
        Finding the extreme lines(border lines)
        '''

        lines = merge_lines(lines, lined_image)

        topedge = [[1000, 1000]]
        bottomedge = [[-1000, -1000]]
        leftedge = [[1000, 1000]]
        leftxintercept = 100000
        rightedge = [[-1000, -1000]]
        rightxintercept = 0
        for i in range(len(lines)):
            current = lines[i][0]
            p = current[0]
            theta = current[1]
            xIntercept = p / np.cos(theta)

            if theta > np.pi * 80 / 180 and theta < np.pi * 100 / 180:
                if p < topedge[0][0]:
                    topedge[0] = current[:]
                if p > bottomedge[0][0]:
                    bottomedge[0] = current[:]

            if theta < np.pi * 10 / 180 or theta > np.pi * 170 / 180:
                if xIntercept > rightxintercept:
                    rightedge[0] = current[:]
                    rightxintercept = xIntercept
                elif xIntercept <= leftxintercept:
                    leftedge[0] = current[:]
                    leftxintercept = xIntercept

        tmpimg = np.copy(lined_image)
        lin_image = np.copy(self.image)
        lin_image = draw_line(leftedge, lin_image)
        lin_image = draw_line(rightedge, lin_image)
        lin_image = draw_line(topedge, lin_image)
        lin_image = draw_line(bottomedge, lin_image)

        tmpimg = draw_line(leftedge, tmpimg)
        tmpimg = draw_line(rightedge, tmpimg)
        tmpimg = draw_line(topedge, tmpimg)
        tmpimg = draw_line(bottomedge, tmpimg)

        return lin_image, tmpimg, leftedge, rightedge, topedge, bottomedge
        

    def calculate_points(self, lin_image, leftedge, rightedge, topedge, bottomedge, height, width):
        '''
        Finding the extreme points 
        '''

        leftedge = leftedge[0]
        rightedge = rightedge[0]
        bottomedge = bottomedge[0]
        topedge = topedge[0]

        left1 = [None, None]
        left2 = [None, None]
        right1 = [None, None]
        right2 = [None, None]
        top1 = [None, None]
        top2 = [None, None]
        bottom1 = [None, None]
        bottom2 = [None, None]

        if leftedge[1] != 0:
            left1[0] = 0
            left1[1] = leftedge[0] / np.sin(leftedge[1])
            left2[0] = width
            left2[1] = -left2[0] / np.tan(leftedge[1]) + left1[1]
        else:
            left1[1] = 0
            left1[0] = leftedge[0] / np.cos(leftedge[1])
            left2[1] = height
            left2[0] = left1[0] - height * np.tan(leftedge[1])

        if rightedge[1] != 0:
            right1[0] = 0
            right1[1] = rightedge[0] / np.sin(rightedge[1])
            right2[0] = width
            right2[1] = -right2[0] / np.tan(rightedge[1]) + right1[1]
        else:
            right1[1] = 0
            right1[0] = rightedge[0] / np.cos(rightedge[1])
            right2[1] = height
            right2[0] = right1[0] - height * np.tan(rightedge[1])

        bottom1[0] = 0
        bottom1[1] = bottomedge[0] / np.sin(bottomedge[1])

        bottom2[0] = width
        bottom2[1] = -bottom2[0] / np.tan(bottomedge[1]) + bottom1[1]

        top1[0] = 0
        top1[1] = topedge[0] / np.sin(topedge[1])
        top2[0] = width
        top2[1] = -top2[0] / np.tan(topedge[1]) + top1[1]

        leftA = left2[1] - left1[1]
        leftB = left1[0] - left2[0]
        leftC = leftA * left1[0] + leftB * left1[1]

        rightA = right2[1] - right1[1]
        rightB = right1[0] - right2[0]
        rightC = rightA * right1[0] + rightB * right1[1]

        topA = top2[1] - top1[1]
        topB = top1[0] - top2[0]
        topC = topA * top1[0] + topB * top1[1]

        bottomA = bottom2[1] - bottom1[1]
        bottomB = bottom1[0] - bottom2[0]
        bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]

        detTopLeft = leftA * topB - leftB * topA

        ptTopLeft = ((topB * leftC - leftB * topC) / detTopLeft, (leftA * topC - topA * leftC) / detTopLeft)

        detTopRight = rightA * topB - rightB * topA

        ptTopRight = ((topB * rightC - rightB * topC) / detTopRight, (rightA * topC - topA * rightC) / detTopRight)

        detBottomRight = rightA * bottomB - rightB * bottomA

        ptBottomRight = ((bottomB * rightC - rightB * bottomC) / detBottomRight, (rightA * bottomC - bottomA * rightC) / detBottomRight)

        detBottomLeft = leftA * bottomB - leftB * bottomA

        ptBottomLeft = ((bottomB * leftC - leftB * bottomC) / detBottomLeft,
                                (leftA * bottomC - bottomA * leftC) / detBottomLeft)

        cv2.circle(lin_image, (int(ptTopLeft[0]), int(ptTopLeft[1])), 5, 0, -1)
        cv2.circle(lin_image, (int(ptTopRight[0]), int(ptTopRight[1])), 5, 0, -1)
        cv2.circle(lin_image, (int(ptBottomLeft[0]), int(ptBottomLeft[1])), 5, 0, -1)
        cv2.circle(lin_image, (int(ptBottomRight[0]), int(ptBottomRight[1])), 5, 0, -1)

        return lin_image, detTopLeft, ptTopLeft, detTopRight, ptTopRight, detBottomRight, ptBottomRight, detBottomLeft, ptBottomLeft


    def find_max_side_len(self, lin_image, detTopLeft, ptTopLeft, detTopRight, ptTopRight, detBottomRight, ptBottomRight, detBottomLeft, ptBottomLeft):
        '''
        Finding the lengths of the maximum sides
        '''

        leftedgelensq = (ptBottomLeft[0] - ptTopLeft[0]) ** 2 + (ptBottomLeft[1] - ptTopLeft[1]) ** 2
        rightedgelensq = (ptBottomRight[0] - ptTopRight[0]) ** 2 + (ptBottomRight[1] - ptTopRight[1]) ** 2
        topedgelensq = (ptTopRight[0] - ptTopLeft[0])**2 + (ptTopLeft[1] - ptTopRight[1]) ** 2
        bottomedgelensq = (ptBottomRight[0] - ptBottomLeft[0]) ** 2 + (ptBottomLeft[1] - ptBottomRight[1]) ** 2
        maxlength = int(max(leftedgelensq, rightedgelensq, bottomedgelensq, topedgelensq) ** 0.5)

        src = [(0, 0)] * 4
        dst = [(0, 0)] * 4
        src[0] = ptTopLeft[:]
        dst[0] = (0, 0)
        src[1] = ptTopRight[:]
        dst[1] = (maxlength - 1, 0)
        src[2] = ptBottomRight[:]
        dst[2] = (maxlength - 1, maxlength - 1)
        src[3] = ptBottomLeft[:]
        dst[3] = (0, maxlength - 1)
        src = np.array(src).astype(np.float32)
        dst = np.array(dst).astype(np.float32)
        self.extracted_grid = cv2.warpPerspective(self.image, cv2.getPerspectiveTransform(src, dst), (maxlength, maxlength))
        self.extracted_grid = cv2.resize(self.extracted_grid, (252, 252))


    def create_image_grid(self):
        '''
        Create a grid on the image(Used for extracting each cell)
        '''

        cells = []

        if self.extracted_grid is None:
            raise Exception('Grid not yet extracted')
        grid = np.copy(self.extracted_grid)
        edge = np.shape(grid)[0]
        celledge = edge // 9

        grid = cv2.bitwise_not(cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1))
        tempgrid = []
        for i in range(celledge, edge+1, celledge):
            for j in range(celledge, edge+1, celledge):
                rows = grid[i-celledge:i]
                tempgrid.append([rows[k][j-celledge:j] for k in range(len(rows))])

        finalgrid = []
        for i in range(0, len(tempgrid)-8, 9):
            finalgrid.append(tempgrid[i:i+9])

        for i in range(9):
            for j in range(9):
                finalgrid[i][j] = np.array(finalgrid[i][j])

        for i in range(9):
            for j in range(9):
                cells.append(finalgrid[i][j])
        
        return np.array(cells)