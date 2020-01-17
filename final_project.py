import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import scipy.spatial
import skimage.draw
import cv2
import time
import face_recognition
from api import PRN
from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box
from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture

'''
    REFERENCES:
    [1]: https://github.com/YadiraF/PRNet
    [2]: https://github.com/wuhuikai/FaceSwap
'''

# PAPER CODE INITIALIZATION
setup_time = time.time()
prn = PRN(is_dlib=True)
print('PAPER CODE SETUP TIME: ', time.time() - setup_time)

# !!!!!!!!!!!!!!!!!!!!!!! GITHUB REPOSITORY CODE FOR POISSON BLENDING START !!!!!!!!!!!!!!!!!!!!!!! REF [2]
def correct_colours(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def apply_mask(img, mask):
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return masked_img


def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask
# !!!!!!!!!!!!!!!!!!!!!!! GITHUB REPOSITORY CODE FOR POISSON BLENDING END !!!!!!!!!!!!!!!!!!!!!!! REF [2]


# FUNCTION MAKES SEGMENTS FOR THE CURRENT IMAGE
def new_portions(image, lines, horizontal_cuts=4, vertical_cuts=1):
    h_increment = len(lines_1) / horizontal_cuts
    test_portion = np.copy(image)

    # CREATE VERTICAL INDEXES FOR CUTS
    v_segments_per_line = []
    for line in range(len(lines)):
        line_length = len(lines[line])
        v_line_increment = line_length / vertical_cuts

        v_line_cuts = []
        for cuts in range(0, vertical_cuts + 1):
            v_cut = cuts * v_line_increment
            if v_cut >= line_length:
                v_cut = line_length - 1
            v_line_cuts.append(int(v_cut))
        v_segments_per_line.append(v_line_cuts)

    # CREATE HORIZONTAL INDEXES FOR CUTS
    line_index_h = []
    for cuts in range(0, horizontal_cuts + 1):
        h_cut = cuts * h_increment
        if h_cut >= len(lines):
            h_cut = len(lines) - 1
        line_index_h.append(int(h_cut))

    # SHOW LINES ON FACE (OPTIONAL)
    for h_line in line_index_h:
        for point in lines[h_line]:
            center = (int(point[0]), int(point[1]))
            thickeness = -1
            point_image = cv2.circle(image, center, 1, (255, 0, 0), 1)
    for v_line in range(len(v_segments_per_line)):
        for vpoint in range(len(v_segments_per_line[v_line])):
            point = lines[v_line][v_segments_per_line[v_line][vpoint]]
            center = (int(point[0]), int(point[1]))
            thickeness = -1
            point_image = cv2.circle(image, center, 1, (0, 0, 255), 1)
    # SHOW IMAGE WITH LINES ACROSS FACE
    #cv2.imshow('CUTS', image)
    #cv2.waitKey(0)

    # MAKE PORTION CUTS FOR THE FACE. MAKES AN ARRAY OF ALL THE PORTIONS WITH THEIR SURROUNDING POINTS
    portion_points = []
    for h_portion in range(len(line_index_h) - 1):
        for v_portion in range(vertical_cuts):
            segment = []
            # TOP LINE
            left_stop = v_segments_per_line[line_index_h[h_portion]][v_portion]
            right_stop = v_segments_per_line[line_index_h[h_portion]][v_portion + 1]
            for point in range(left_stop, right_stop + 1):
                segment.append(lines[line_index_h[h_portion]][point])
            # BOTTOM LINE
            left_stop = v_segments_per_line[line_index_h[h_portion + 1]][v_portion]
            right_stop = v_segments_per_line[line_index_h[h_portion + 1]][v_portion + 1]
            for point in range(left_stop, right_stop + 1):
                segment.append(lines[line_index_h[h_portion + 1]][point])
            # LEFT & RIGHT SIDE
            for line in range(line_index_h[h_portion], line_index_h[h_portion + 1]):
                left_stop = v_segments_per_line[line][v_portion]
                right_stop = v_segments_per_line[line][v_portion + 1]
                segment.append(lines[line][left_stop])
                segment.append(lines[line][right_stop])
            portion_points.append(segment)

    # RETURN ALL THE PORTIONS OF THE FACE
    return portion_points


# FUNCTION GATHERS VERTICES OF THE FACE, PROCESSES THESE INTO ROWS, AND THEN PROCESSES TO REMOVE SOME OBFUSCATION
def main(image_path, cropped_lines=None):

    # GET THE DENSE AND SPECIFIC LANDMARKS USING PAPER CODE
    # !!!!!!!!!!!!!!!!!!!!!!! PAPER CODE START !!!!!!!!!!!!!!!!!!!!!!! REF [1]
    image = imread(image_path)
    [h, w, c] = image.shape
    if c > 3:
        image = image[:, :, :3]

    max_size = max(image.shape[0], image.shape[1])
    if max_size > 1000:
        image = rescale(image, 1000. / max_size)
        image = (image * 255).astype(np.uint8)

    paper_time = time.time()
    pos = prn.process(image)  # use dlib to detect face

    image = image / 255.

    vertices = prn.get_vertices(pos)
    kpt = prn.get_landmarks(pos)[:, :2]
    face = np.zeros(image.shape)
    vertx = vertices[:, :2]
    print('TIME FOR REFERENCE TAKEN BY PAPER CODE: ', time.time() - paper_time)
    # !!!!!!!!!!!!!!!!!!!!!!! PAPER CODE END !!!!!!!!!!!!!!!!!!!!!!! REF [1]

    # DATA STRUCTURE FOR LINES
    lines = []
    line = []
    break_points = []

    # GO THROUGH ALL THE VERTICES OF THE FACE
    for i in range(len(vertx)):
        if len(line) == 0:
            line.append(vertx[i])
        else:
            # IF THE POINTS ARE VERY FAR IN DISTANCE IN THE X AXIS
            if abs(vertx[i][0] - vertx[i-1][0]) > 50 or abs(vertx[i][1] - vertx[i-1][1]) > 50:
                # ADD THE LINE TO THE NEW STRUCTURE AND CREATE AN EMPTY LINE
                break_points.append(vertx[i - 1])
                break_points.append(vertx[i])
                lines.append(line)
                line = []
                line.append(vertx[i])
        line.append(vertx[i])

    # UNUSED FUNCTION CURRENTLY TO DISPLAY ALL THE DIFFERENT FACIAL LINES WITH DIFFERENT COLORS
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    index = 0
    point_image = np.copy(image)
    for cur_line in lines:
        for point in cur_line:
            center = (int(point[0]), int(point[1]))
            thickeness = -1
            point_image = cv2.circle(point_image, center, 1, color[index], 0)
        index += 1
        if index == 3:
            index = 0
    #cv2.imshow('', point_image)
    #cv2.waitKey(0)

    # CREATE DATA STRUCTURES
    lines_cropped = []
    lines_cropped_index = []
    # CHECK IF THERE IS A RELATED INDEX TO USE. IF SO UTILIZE THAT ONE.
    # THIS IS TO USE ON THE SOURCE IMAGE AFTER PROCESSING THE DESTINATION IMAGE
    if cropped_lines is None:
        # GO THROUGH ALL THE LINES
        for cur_line in range(len(lines)):
            left_index = 0
            right_index = len(lines[cur_line]) - 1
            highest_left = 1000000
            highest_right = 0
            # GO THROUGH ALL POINTS AND GET THE FARTHEST LEFT AND RIGHT INDEX
            for cur_point in range(len(lines[cur_line])):
                if cur_point != 0 and cur_point != len(lines[cur_line]) - 1:
                    if lines[cur_line][cur_point][0] < highest_left:
                        highest_left = lines[cur_line][cur_point][0]
                        left_index = cur_point
                    elif lines[cur_line][cur_point][0] > highest_right:
                        highest_right = lines[cur_line][cur_point][0]
                        right_index = cur_point

            # USED TO REMOVE PARTS OF THE SIDE OF THE FACE TO REMOVE TEARING
            outside_removal_amount = int(len(lines[cur_line]) * .1)
            if left_index < 0 + outside_removal_amount:
                left_index = left_index + outside_removal_amount
            if right_index > len(lines[cur_line]) - 1 - outside_removal_amount:
                right_index = right_index - outside_removal_amount

            # APPEND THE POINTS BETWEEN THE LEFT AND RIGHT MOST INDEX
            lines_cropped.append(lines[cur_line][left_index:right_index])
            lines_cropped_index.append([left_index, right_index])
    else:
        # GO THROUGH ALL THE LINES FOR THE REFERENCE AND JUST MATCH THEIR LENGTH
        for cur_line in range(len(cropped_lines)):
            lines_cropped.append(lines[cur_line][cropped_lines[cur_line][0]:cropped_lines[cur_line][1]])

    # RETURN FACE LINES, FACIAL LOCATIONS DEFINED, THE CROPPED LINES AND THEIR START AND END INDEXES
    return lines, kpt, lines_cropped, lines_cropped_index


total_algorithm = time.time()
# INPUT FOR PHOTOS
# DST_IMAGE WILL BE THE IMAGE THAT SRC_IMAGE FACE WILL BE PLACED ON
dst_image = 'Input/girl.jpg'
src_image = 'Input/face.jpg'

# BETTER TO BASE LINES OFF OF WORST RESULTS, BUT HAVE TO COMPUTE BOTH FIRST SO CAUSES EXPENSE.
# ALSO DEFINITION OF WORSE RESULTS IS SUBJECTIVE. UTILIZE WORST OF BOTH PROBABLY

lines_1, landmarks_1, lines_cropped_1, lines_cropped_index_1 = main(dst_image)
lines_2, landmarks_2, lines_cropped_2, lines_cropped_index_2 = main(src_image, lines_cropped_index_1)

# CHANGING LINES, THIS IS A TEST TO TRY AND STOP TEARING, WORKS FOR GIRL ANGLED
lines_1 = lines_cropped_1
lines_2 = lines_cropped_2

im1 = cv2.imread(dst_image)
im2 = cv2.imread(src_image)

# !!!!!!!!!!!!!!!!!! TRIAL STUFF !!!!!!!!!!!!!!!!!!
lines = np.asarray(lines_2)
image = np.copy(im2)

# 60, 80 for both works for girl angled
horizontal_sections = 20
vertical_sections = 40
total_sections = horizontal_sections * vertical_sections

# USE THESE FOR NO LINES ON FINAL IMAGE
im1_portion_points = new_portions(np.copy(im1), lines_1, horizontal_sections, vertical_sections)
im2_portion_points = new_portions(np.copy(im2), lines_2, horizontal_sections, vertical_sections)

# HOMOGRAPHY TIME
projection_start = time.time()

# STARTING PROJECTION ON THE FACIAL SEGEMENTS
final_image = np.copy(im1)
final_image_blend = np.copy(im1)
just_face = np.zeros(im1.shape, dtype=np.uint8)
size = im1.shape
width, height, channels = im1.shape
center = (int(height/2), int(width/2))

incorrect_maps = 0
warping_image = np.copy(im2)

# GO THROUGH ALL THE SECTIONS OF THE FACE
for portions in range(len(im1_portion_points)):
    # ORIGINAL SOURCE PORTION TO TRY AND IMPLEMENT ENTROPY
    points_used = np.asarray(im2_portion_points[portions])
    vertices = scipy.spatial.ConvexHull(points_used).vertices
    Y_src, X_src = skimage.draw.polygon(points_used[vertices, 1], points_used[vertices, 0])

    # PORTION OF DESTINATION IMAGE
    test_image = np.zeros(im1.shape, dtype=np.uint8)
    points_used = np.asarray(im1_portion_points[portions])
    vertices = scipy.spatial.ConvexHull(points_used).vertices
    Y, X = skimage.draw.polygon(points_used[vertices, 1], points_used[vertices, 0])
    test_image[Y, X] = [255, 255, 255]

    num_pixels = len(Y)
    threshold = 1800 * num_pixels

    # POINTS FROM THE CURRENT SECTION ON SOURCE AND DESTINATION IMAGE
    test_points_im1 = np.asarray(im1_portion_points[portions])
    test_points_im2 = np.asarray(im2_portion_points[portions])

    # DEFINE PROJECTIVE MATRIX AND WARP THE IMAGE
    try:
        h, status = cv2.findHomography(test_points_im2, test_points_im1, cv2.RANSAC, 5.0)
        im_warp = cv2.warpPerspective(warping_image, h, (size[1], size[0]))
    except:
        continue

    # SAVE STATE FOR PREVIOUS BACKUP
    save_last = np.copy(final_image)
    save_face = np.copy(just_face)

    # GRAB JUST THE WANTED SECTION OF THE WARPED IMAGE
    portion = np.copy(im_warp[Y, X])

    # THROW AWAY PORTIONS THAT DO NOT WORK, AND DEFINE VARIABLES FOR FUTURE USAGE
    try:
        mean_src = [int(np.mean(im2[Y_src, X_src][0])), int(np.mean(im2[Y_src, X_src][1])), int(np.mean(im2[Y_src, X_src][2]))]
        mean_warped = [int(np.mean(im2[Y, X][0])), int(np.mean(im2[Y, X][1])), int(np.mean(im2[Y, X][2]))]
    except:
        mean_src = [0, 0, 0]
        mean_warped = [100, 100, 100]

    # SET PORTION INTO PART OF THE FINAL IMAGES
    final_image[Y, X] = portion
    just_face[Y, X] = portion

    change = np.sum(abs(final_image - save_last))

    # TRYING TO STOP INCORRECT TRANSFORMATION, WOULD GO BACK TO PREVIOUS FACE IF IT DOESNT WORK
    threshold_percentage = 8
    if change > threshold:  # or abs(sum(mean_src) - sum(mean_warped)) > threshold_percentage*sum(mean_src):
        incorrect_maps += 1
        final_image = save_last
        just_face = save_face


# OUTPUT TIME AND RESULTS FROM TRANSFORMATION
print('TIME / SECTION: ', (time.time() - projection_start) / (vertical_sections * horizontal_sections))
print('WARP SECTIONS THROWN OUT: ', incorrect_maps/total_sections*100, "%")

# REMOVE LIPS FROM THE FINAL SOURCE FACE
lips = landmarks_1[48:62]
vertices = scipy.spatial.ConvexHull(lips).vertices
Y_lips, X_lips = skimage.draw.polygon(lips[vertices, 1], lips[vertices, 0])
just_face[Y_lips, X_lips] = [0, 0, 0]

# FACE PORTION FROM THE DESTINATION FACE
im1_portion_points = new_portions(np.copy(im1), lines_1, 1, 1)
test_image = np.zeros(im1.shape, dtype=np.uint8)
test_mask = np.zeros(im1.shape[:2], dtype=np.uint8)
points_used = np.asarray(im1_portion_points[0])
vertices = scipy.spatial.ConvexHull(points_used).vertices
Y, X = skimage.draw.polygon(points_used[vertices, 1], points_used[vertices, 0])
y_min = min(Y)
y_max = max(Y)
x_min = min(X)
x_max = max(X)

# DEFINE CENTER PLACEMENT FOR FACE
center_new = (int((x_max+x_min)/2), int((y_max+y_min)/2))

# GET JUST THE DESTINATION FACE
test_image[Y, X] = im1[Y, X]
test_mask[Y, X] = 255
kernel = np.ones((10, 10), dtype=np.uint8)
test_mask = cv2.erode(test_mask, kernel, iterations=1)

# BLEND FACE AND THEN SHOW AND WRITE THE OUTPUT IMAGE
# !!!!!!!!!!!!!!!!!!!!!!! UTILIZING POISSON BLURRING FROM OTHER SOURCE !!!!!!!!!!!!!!!!!!!!!!! REF [2]
mask_src = np.mean(just_face, axis=2) > 0
mask = np.asarray(test_mask*mask_src, dtype=np.uint8)

warped_src = np.copy(just_face)
warped_src = apply_mask(warped_src, mask)
test_image = apply_mask(test_image, mask)
warped_src = correct_colours(test_image, warped_src, landmarks_1)

cv2.imshow('warped dest', test_image)
cv2.imshow('warped source', warped_src)
cv2.waitKey(0)

mask = cv2.erode(mask, kernel, iterations=1)

center_placement = np.asarray(landmarks_1[33])
center_placement = (int(center_placement[0]), int(center_placement[1]))

width, height, channels = im1.shape
center = (int(height/2), int(width/2))

output = cv2.seamlessClone(warped_src, im1, mask, center_new, cv2.NORMAL_CLONE)

cv2.imshow('Blended Output', output)
cv2.imwrite('Input/output2.jpg', output)
cv2.waitKey(0)
# !!!!!!!!!!!!!!!!!!!!!!! END OF POISSON BLURRING AND OUTPUT !!!!!!!!!!!!!!!!!!!!!!! REF [2]

print('TIME FOR FULL ALGORITHM: ', time.time() - total_algorithm)






