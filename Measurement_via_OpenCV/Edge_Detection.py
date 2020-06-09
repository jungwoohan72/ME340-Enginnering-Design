# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math

#이미지를 물체 크기에 맞게 대략적으로 Crop 하기
img = cv2.imread("prototype5.jpeg")
crop_img = img[500:950, 900:1500]
#crop_img = img[200:600, 400:1100]
cv2.imshow("crop_img", crop_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Morphology 이용해서 노이즈 최대한 제거
for i in range(30):
  kernel = np.ones((50,50), np.uint8) # 필터 사이즈 10x10
  crop_img_morph = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel)

#Contour Detection을 위해 그레이 스케일로 이미지 변환
grayscale = cv2.cvtColor(crop_img_morph, cv2.COLOR_BGR2GRAY)

#위에서 변환한 Gray Scale 이미지를 Input으로 받아 픽셀의 색깔이 170보다 크면 흰색으로 변환
thresholdTuple = cv2.threshold(grayscale, 180, 255, cv2.THRESH_BINARY)
blackwhite = thresholdTuple[1] #바이너리 이미지
contour, hierarchy = cv2.findContours(blackwhite, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #흰색과 검은색을 경계로 하는 Contour 좌표들 추출
cv2.imshow('blackwhite', blackwhite)

#물체에 외접하는 가장 작은 사각형 구하기
bounding_rect = crop_img_morph.copy()
cnt = contour[0]
x, y, w, h = cv2.boundingRect(cnt) #물체에 외접하는 가장 작은 사각형의 왼쪽 위 좌표와 너비와 높이 길이
cv2.rectangle(bounding_rect,(x,y),(x+w,y+h),(0,0,255),2) #위 좌표에 상응하는 사각형 그리기
#cv2.imshow("bounding_rect", bounding_rect)

#Contour 좌표들을 왼변, 아랫변, 우변, 윗변 순서대로 나열하는 작업
contour_separate = crop_img_morph.copy()
sorted_ctrs = sorted(contour, key=lambda ctr: x + y * blackwhite.shape[1]) #왼쪽 위 -> 왼쪽 아래 -> 오른쪽 아래 -> 오른쪽 위 -> 왼쪽 위

#좌표를 상하좌우로 나누기 위한 Edge Detection
whole_corner = crop_img_morph.copy()
whole_corner_gray = cv2.cvtColor(whole_corner,cv2.COLOR_BGR2GRAY)

whole_corners = cv2.goodFeaturesToTrack(whole_corner_gray,15,0.20,15)
whole_corners = np.int0(whole_corners)
whole_sorted_corners = sorted(whole_corners, key=lambda cnr: cnr[0][0]) # x좌표를 기준으로 코너들 정렬

for i in whole_corners:
    a,b = i.ravel()
    cv2.circle(whole_corner,(a,b),3,(0,0,255),-1)

cv2.imshow("whole_corner",whole_corner)


#좌표들 상하좌우로 나누기
left = []
bottom = []
right = []
top = []

for coordinate in sorted_ctrs[0]:
    if coordinate[0][0] >= whole_sorted_corners[1][0][0] and coordinate[0][1] >= whole_sorted_corners[1][0][1]:
        bottom.append(coordinate)
    elif coordinate[0][0] <= whole_sorted_corners[1][0][0] and coordinate[0][1] > whole_sorted_corners[0][0][1]:
        left.append(coordinate)
    elif coordinate[0][0] >= whole_sorted_corners[-2][0][0] and coordinate[0][1] > whole_sorted_corners[-2][0][1]:
        right.append(coordinate)
    else:
        top.append(coordinate)

#나눈 좌표들 색깔 별로 Plot 하기
cv2.drawContours(contour_separate, left, -1, (255,0,0),2)
cv2.drawContours(contour_separate, bottom, -1, (0,255,0),2)
cv2.drawContours(contour_separate, right, -1, (0,0,255),2)
cv2.drawContours(contour_separate, top, -1, (0,255,255),2)
cv2.imshow('contour_separate', contour_separate)

#Pre-processing top list
corner = crop_img_morph.copy()
corner_top = corner[0:int(crop_img_morph.shape[1]/2),:] #윗변을 제외한 나머지 좌표들은 필요 없으므로 일단은 코너 디텍션에서 제외
corner_gray = cv2.cvtColor(corner_top,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(corner_gray,15,0.20,15)
corners = np.int0(corners)

top_copy = crop_img_morph.copy()
sorted_corners = sorted(corners, key=lambda cnr: cnr[0][0]) # x좌표를 기준으로 코너들 정렬


top_left_straight = []
top_left_gradient = []
top_center_left = []
top_center = []
top_center_right = []
top_right_gradient = []
top_right_straight = []

for coordinate in top:
    if coordinate[0][1] < crop_img.shape[0]/2 and coordinate[0][0] > sorted_corners[0][0][0] and coordinate[0][0] < sorted_corners[1][0][0]:
        top_left_straight.append(coordinate)
    elif coordinate[0][1] < sorted_corners[2][0][1] and coordinate[0][0] > sorted_corners[1][0][0] and coordinate[0][0] < sorted_corners[2][0][0]:
        top_left_gradient.append(coordinate)
    elif coordinate[0][1] > sorted_corners[2][0][1] and coordinate[0][1] < sorted_corners[3][0][1] and coordinate[0][0] > sorted_corners[2][0][0]-5 and coordinate[0][0] < sorted_corners[2][0][0]+5:
        top_center_left.append(coordinate)
    elif coordinate[0][1] < crop_img.shape[0]/2 and coordinate[0][0] > sorted_corners[3][0][0] and coordinate[0][0] < sorted_corners[4][0][0]:
        top_center.append(coordinate)
    elif coordinate[0][1] < sorted_corners[4][0][1] and coordinate[0][1] > sorted_corners[5][0][1] and coordinate[0][0] > sorted_corners[4][0][0]-5 and coordinate[0][0] < sorted_corners[4][0][0]+5:
        top_center_right.append(coordinate)
    elif coordinate[0][1] < sorted_corners[5][0][1] and coordinate[0][0] > sorted_corners[5][0][0] and coordinate[0][0] < sorted_corners[6][0][0]:
        top_right_gradient.append(coordinate)
    elif coordinate[0][1] < crop_img.shape[0]/2 and coordinate[0][0] > sorted_corners[6][0][0] and coordinate[0][0] < sorted_corners[7][0][0]:
        top_right_straight.append(coordinate)

cv2.drawContours(top_copy, top_left_straight, -1, (0,0,255),2)
cv2.drawContours(top_copy, top_left_gradient, -1, (0,255,0),2)
cv2.drawContours(top_copy, top_center_left, -1, (255,0,0),2)
cv2.drawContours(top_copy, top_center, -1, (0,255,0),2)
cv2.drawContours(top_copy, top_center_right, -1, (255,0,0),2)
cv2.drawContours(top_copy, top_right_gradient, -1, (0,255,0),2)
cv2.drawContours(top_copy, top_right_straight, -1, (0,0,255),2)
cv2.imshow("top_copy", top_copy)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#채점기준 1 : 길이
#각 변의 y좌표 평균 구하는 함수
def get_average_y_coordinate(sep_contrs):
    y_sum = 0
    for coordinate in sep_contrs:
        y_sum += coordinate[0][1]
    average_y = y_sum/len(sep_contrs)
    return average_y

#왼변과 오른변의 x좌표 구하기
left_x_sum = 0
right_x_sum = 0

for coordinate in left:
    left_x_sum += coordinate[0][0]
for coordinate in right:
    right_x_sum += coordinate[0][0]

left_x = left_x_sum/len(left)
right_x = right_x_sum/len(right)

bottom_width = right_x - left_x #밑변의 너비

top_left_straight_y = get_average_y_coordinate(top_left_straight)
top_right_straight_y = get_average_y_coordinate(top_right_straight)
bottom_y = get_average_y_coordinate(bottom)

left_height = bottom_y - top_left_straight_y #왼변의 높이
right_height = bottom_y - top_right_straight_y #오른변의 높이

top_left_straight_width = sorted_corners[1][0][0] - sorted_corners[0][0][0]
top_left_gradient_length = int(math.sqrt((sorted_corners[2][0][0] - sorted_corners[1][0][0])**2 + (sorted_corners[2][0][1] - sorted_corners[1][0][1])**2))
top_center_left_height = sorted_corners[3][0][1] - sorted_corners[2][0][1]
top_center_width = sorted_corners[4][0][0] - sorted_corners[3][0][0]
top_center_right_height = sorted_corners[4][0][1] - sorted_corners[5][0][1]
top_right_gradient_length = int(math.sqrt((sorted_corners[6][0][0] - sorted_corners[5][0][0])**2 + (sorted_corners[5][0][1] - sorted_corners[6][0][1])**2))
top_right_straight_width = sorted_corners[7][0][0] - sorted_corners[6][0][0]

print("밑변의 길이: %d, 좌변의 길이: %d, 우변의 길이: %d\n" % (bottom_width, left_height, right_height))
print("윗면 각 부분의 길이: %d, %d, %d, %d, %d, %d, %d\n " % (top_left_straight_width, top_left_gradient_length, top_center_left_height, top_center_width, top_center_right_height, top_right_gradient_length, top_right_straight_width))

#채점 기준 2 : 대칭

left_right = abs(left_height-right_height)
top_left_right_straight = abs(top_left_straight_width-top_right_straight_width)
top_left_right_gradient = abs(top_left_gradient_length-top_right_gradient_length)
top_left_straight_center = abs(top_center_left_height-top_center_right_height)

bottom_center = (left_x + right_x)/2
top_center_left_half = bottom_center - sorted_corners[3][0][0]
top_center_right_half = sorted_corners[4][0][0] - bottom_center

print("밑변의 중심: %d, 밑변의 중심으로부터 왼쪽까지: %d, 밑변의 중심으로부터 오른쪽까지: %d\n" % (bottom_center, top_center_left_half, top_center_right_half))

#채점 기준 3 : 빗면 각도
left_gradient_angle_lst = []
for coordinate in top_left_gradient:
    for i in range(len(top_left_gradient)):
        if top_left_gradient[i][0][0] == coordinate[0][0] and top_left_gradient[i][0][1] == coordinate[0][1]:
            continue
        else:
            angle = math.atan2(top_left_gradient[i][0][1] - coordinate[0][1], top_left_gradient[i][0][0] - coordinate[0][0]) #모든 좌표들 간의 각도 평균을 구한다.
            left_gradient_angle_lst.append(angle)
left_gradient_angle_avg_rad = sum(left_gradient_angle_lst)/len(left_gradient_angle_lst)
left_gradient_angle_avg_deg = float(abs(left_gradient_angle_avg_rad)*180/math.pi)

right_gradient_angle_lst = []
for coordinate in top_right_gradient:
    for i in range(len(top_right_gradient)):
        if top_right_gradient[i][0][0] == coordinate[0][0] and top_left_gradient[i][0][1] == coordinate[0][1]:
            continue
        else:
            angle = math.atan2(coordinate[0][1] - top_right_gradient[i][0][1], top_right_gradient[i][0][0] - coordinate[0][0]) #모든 좌표들 간의 각도 평균을 구한다.

            right_gradient_angle_lst.append(angle)
right_gradient_angle_avg_rad = sum(right_gradient_angle_lst)/len(right_gradient_angle_lst)
right_gradient_angle_avg_deg = float(abs(right_gradient_angle_avg_rad)*180/math.pi)

print("오른쪽 빗면의 각도: %f, 왼쪽 빗면의 각도: %f\n" % (left_gradient_angle_avg_deg, right_gradient_angle_avg_deg))

#채점 기준 4 : 빗면으로부터 좌표까지의 최단거리
#Shortest Distacane from a point to a line
def norm_distance(p1, p2, input_list):
    norm_distance_list = []
    for p3 in input_list:
        d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
        norm_distance_list.append(d)
    norm_distance_avg = sum(norm_distance_list)/len(norm_distance_list)
    return norm_distance_avg

left_gradient_norm_distnace = norm_distance(sorted_corners[1][0], sorted_corners[2][0], top_left_gradient)
right_gradient_norm_distnace = norm_distance(sorted_corners[5][0], sorted_corners[6][0], top_right_gradient)

print("각 경사 그래프로부터 좌표까지의 최단거리 평균")
print("왼쪽 경사: %f, 오른쪽 경사: %f\n" % (left_gradient_norm_distnace[0], right_gradient_norm_distnace[0]))

#채점 기준 5 : 기울기

def slope(contour_list):
    angle_lst = []
    for i in range(len(contour_list)//2):
        if contour_list == top_left_straight:
            angle = math.atan2(contour_list[i][0][1] - contour_list[-1-i][0][1], contour_list[i][0][0] - contour_list[-1-i][0][0])
            angle_lst.append(angle)
        else:
            angle = math.atan2(contour_list[-1-i][0][1] - contour_list[i][0][1], contour_list[-1-i][0][0] - contour_list[i][0][0]) #변을 10등분해서 그 나눠진 변들의 각도를 구한다.
            angle_lst.append(angle)
    angle_avg_rad = sum(angle_lst)/len(angle_lst)
    angle_avg_deg = abs(angle_avg_rad)*180/math.pi
    return float(angle_avg_deg)

print(slope(left))
print(slope(bottom))
print(slope(right))
print(slope(top_left_straight))
print(slope(top_center_left))
print(slope(top_center))
print(slope(top_center_right))
print(slope(top_right_straight))

#채점 기준 6 : 표면 마무리
#Standard deviation of x,y coordinates
def x_std(input_list):
    x_list = []
    for coordinate in input_list:
        x_list.append(coordinate[0][0])
    x_std = np.std(x_list)
    return x_std

def y_std(input_list):
    y_list = []
    for coordinate in input_list:
        y_list.append(coordinate[0][1])
    y_std = np.std(y_list)
    return y_std

left_std = x_std(left)
bottom_std = y_std(bottom)
right_std = x_std(right)
top_left_straight_std = y_std(top_left_straight)
top_center_left_std = x_std(top_center_left)
top_center_std = y_std(top_center)
top_center_right_std = x_std(top_center_right)
top_right_straight_std = y_std(top_right_straight)

print("수평이거나 수직인 변 좌표들의 표준편차")
print("좌변: %f, 밑변: %f, 우변: %f" % (left_std, bottom_std, right_std))
print("윗면: %f, %f, %f, %f, %f" % (top_left_straight_std, top_center_left_std, top_center_std, top_center_right_std, top_right_straight_std))

#Desplaying on screen

display_dimension_img = crop_img_morph.copy()
display_slope_img = crop_img_morph.copy()
display_std_img = crop_img_morph.copy()
parts = [top_left_straight, top_left_gradient, top_center_left, top_center, top_center_right, top_right_gradient, top_right_straight, right, bottom, left]
part_dimension_pixel = [top_left_straight_width, top_left_gradient_length, top_center_left_height, top_center_width, top_center_right_height, top_right_gradient_length, top_right_straight_width, right_height, bottom_width, left_height]
part_slope = [round(slope(top_left_straight),2), round(left_gradient_angle_avg_deg,2), round(slope(top_center_left),2), round(slope(top_center),2), round(slope(top_center_right),2), round(right_gradient_angle_avg_deg,2), round(slope(top_right_straight),2),round(slope(right),2),round(slope(bottom),2),round(slope(left),2)]
part_dimension_real = []
for part in part_dimension_pixel:
    part_dimension_real.append(round(float(part)*0.08898,3))

font = cv2.FONT_HERSHEY_SIMPLEX   # hand-writing style font
fontScale = 0.5
thickness = 1
for i in range(len(parts)):
    if parts[i] == top_center_left:
        location = (parts[i][len(parts[i])//2][0][0]-50, parts[i][len(parts[i])//2][0][1])
        cv2.putText(display_dimension_img, str(part_dimension_real[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == top_center:
        location = (parts[i][len(parts[i])//2][0][0]-20, parts[i][len(parts[i])//2][0][1]+20)
        cv2.putText(display_dimension_img, str(part_dimension_real[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == top_left_gradient:
        location = (parts[i][len(parts[i])//2][0][0]-60, parts[i][len(parts[i])//2][0][1])
        cv2.putText(display_dimension_img, str(part_dimension_real[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == top_left_straight:
        location = (parts[i][len(parts[i])//2][0][0] -20, parts[i][len(parts[i])//2][0][1]-10)
        cv2.putText(display_dimension_img, str(part_dimension_real[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == right:
        location = (parts[i][len(parts[i])//2][0][0] -60, parts[i][len(parts[i])//2][0][1])
        cv2.putText(display_dimension_img, str(part_dimension_real[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == bottom:
        location = (parts[i][len(parts[i])//2][0][0]-30, parts[i][len(parts[i])//2][0][1]-10)
        cv2.putText(display_dimension_img, str(part_dimension_real[i]), location, font, fontScale, (0,0,255), thickness)
    else:
        location = (parts[i][len(parts[i])//2][0][0], parts[i][len(parts[i])//2][0][1])
        cv2.putText(display_dimension_img, str(part_dimension_real[i]), location, font, fontScale, (0,0,255), thickness)

cv2.imshow("dimension", display_dimension_img)

for i in range(len(parts)):
    if parts[i] == top_center_left:
        location = (parts[i][len(parts[i])//2][0][0]-50, parts[i][len(parts[i])//2][0][1])
        cv2.putText(display_slope_img, str(part_slope[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == top_center:
        location = (parts[i][len(parts[i])//2][0][0]-20, parts[i][len(parts[i])//2][0][1]+20)
        cv2.putText(display_slope_img, str(part_slope[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == top_left_gradient:
        location = (parts[i][len(parts[i])//2][0][0]-60, parts[i][len(parts[i])//2][0][1])
        cv2.putText(display_slope_img, str(part_slope[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == top_left_straight:
        location = (parts[i][len(parts[i])//2][0][0] -20, parts[i][len(parts[i])//2][0][1]-10)
        cv2.putText(display_slope_img, str(part_slope[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == right:
        location = (parts[i][len(parts[i])//2][0][0] -60, parts[i][len(parts[i])//2][0][1])
        cv2.putText(display_slope_img, str(part_slope[i]), location, font, fontScale, (0,0,255), thickness)
    elif parts[i] == bottom:
        location = (parts[i][len(parts[i])//2][0][0]-30, parts[i][len(parts[i])//2][0][1]-10)
        cv2.putText(display_slope_img, str(part_slope[i]), location, font, fontScale, (0,0,255), thickness)
    else:
        location = (parts[i][len(parts[i])//2][0][0], parts[i][len(parts[i])//2][0][1])
        cv2.putText(display_slope_img, str(part_slope[i]), location, font, fontScale, (0,0,255), thickness)

cv2.imshow("slope", display_slope_img)

part_std = [top_left_straight, top_center_left, top_center, top_center_right, top_right_straight, right, bottom, left]
part_pixel_std = [top_left_straight_std,top_center_left_std, top_center_std, top_center_right_std, top_right_straight_std, right_std, bottom_std,left_std]
part_real_std = []
for part in part_pixel_std:
    part_real_std.append(round(float(part)*0.08898,3))

for i in range(len(part_std)):
    if part_std[i] == top_center_left:
        location = (part_std[i][len(part_std[i])//2][0][0]-50, part_std[i][len(part_std[i])//2][0][1])
        cv2.putText(display_std_img, str(part_real_std[i]), location, font, fontScale, (0,0,255), thickness)
    elif part_std[i] == top_center:
        location = (part_std[i][len(part_std[i])//2][0][0]-20, part_std[i][len(part_std[i])//2][0][1]+20)
        cv2.putText(display_std_img, str(part_real_std[i]), location, font, fontScale, (0,0,255), thickness)
    elif part_std[i] == top_left_gradient:
        location = (part_std[i][len(part_std[i])//2][0][0]-60, part_std[i][len(part_std[i])//2][0][1])
        cv2.putText(display_std_img, str(part_real_std[i]), location, font, fontScale, (0,0,255), thickness)
    elif part_std[i] == top_left_straight:
        location = (part_std[i][len(part_std[i])//2][0][0] -20, part_std[i][len(part_std[i])//2][0][1]-10)
        cv2.putText(display_std_img, str(part_real_std[i]), location, font, fontScale, (0,0,255), thickness)
    elif part_std[i] == right:
        location = (part_std[i][len(part_std[i])//2][0][0] -60, part_std[i][len(part_std[i])//2][0][1])
        cv2.putText(display_std_img, str(part_real_std[i]), location, font, fontScale, (0,0,255), thickness)
    elif part_std[i] == bottom:
        location = (part_std[i][len(part_std[i])//2][0][0]-30, part_std[i][len(part_std[i])//2][0][1]-10)
        cv2.putText(display_std_img, str(part_real_std[i]), location, font, fontScale, (0,0,255), thickness)
    else:
        location = (part_std[i][len(part_std[i])//2][0][0], part_std[i][len(part_std[i])//2][0][1])
        cv2.putText(display_std_img, str(part_real_std[i]), location, font, fontScale, (0,0,255), thickness)

cv2.imshow("std", display_std_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
