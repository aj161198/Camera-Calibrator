from __future__ import division
from util import *
from darknet import Darknet
from calibutils import *
import cv2
import numpy as np
import argparse
import pickle as pkl
import matplotlib.pyplot as plt


def arg_parse():
    parser = argparse.ArgumentParser(description="Camera Calibration from traffic scene")
    parser.add_argument("-video", dest="video", help="Video to Calibrate from.")
    parser.add_argument("--display", dest="display", help="Display trajectories for intrinsic calibration")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.8)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()


def get_thresholds(angles, mags):
    bins, z, _ = plt.hist(angles)
    if len(np.where(bins == np.max(bins))[0]) > 1:
        lower_angle_thresh, upper_angle_thresh = z[np.where(bins == np.max(bins))[0][0]], z[
            np.where(bins == np.max(bins))[0][1]]
    else:
        lower_angle_thresh, upper_angle_thresh = z[np.where(bins == np.max(bins))], z[
            np.where(bins == np.max(bins))[0] + 1]
    plt.close("all")

    bins, z, _ = plt.hist(mags)
    if len(np.where(bins == np.max(bins))[0]) > 1:
        lower_mag_thresh = z[np.where(bins == np.max(bins))[0][0]]
        upper_mag_thresh = z[np.where(bins == np.max(bins))[0][1]]
    else:
        if np.where(bins == np.max(bins))[0] - 1 >= 0:
            lower_mag_thresh = z[np.where(bins == np.max(bins))[0] - 1]
        else:
            lower_mag_thresh = z[np.where(bins == np.max(bins))[0]]

        if np.where(bins == np.max(bins))[0] + 1 <= len(z) - 1:
            upper_mag_thresh = z[np.where(bins == np.max(bins))[0] + 1]
        else:
            upper_mag_thresh = z[np.where(bins == np.max(bins))[0]]
    return lower_angle_thresh, upper_angle_thresh, lower_mag_thresh, upper_mag_thresh


cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

args = arg_parse()
video_file_name = args.video
coordinates, visibility_indices, keypoints_indices = get_trajectories("sample_video/" + video_file_name)

cap = cv2.VideoCapture("sample_video/" + video_file_name)
ret, frame = cap.read()
print(ret, "sample_video/" + video_file_name)
h, w, c = frame.shape
display = args.display
x_distorted, y_distorted = post_process_trajectories(frame, coordinates, visibility_indices, keypoints_indices, display)

min_error = 100
diagonal = (h * h + w * w) ** 0.5
min_factor = 0
for factor in range(0, 300):
    f = diagonal / ((factor + 1) / 100)
    K_estimate = np.array([[f, 0., w / 2], [0., f, h / 2], [0., 0., 1.]])
    x_undistorted, y_undistorted = undistortPoints(x_distorted, y_distorted, K_estimate)
    error = 0
    for i in range(len(keypoints_indices)):
        lower = np.logical_and(np.greater_equal(x_undistorted[i, :], 0), np.greater_equal(y_undistorted[i, :], 0))
        upper = np.logical_and(np.less(x_undistorted[i, :], w), np.less(y_undistorted[i, :], h))
        within_indices = np.logical_and(lower, upper)
        xs, ys = x_undistorted[i, :][within_indices], y_undistorted[i, :][within_indices]
        if xs.shape[0] > 20:
            error = error + get_line(xs, ys)[-1]

    if error <= min_error:
        min_error = error
        K = K_estimate
        min_factor = factor

print(K, min_factor)

D = computeDistortionCoefficients(K)
print(D)

# ---------------------Extrinsic Calibration-------------------------------------------------------------------------- #

# Model Initialization
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
CUDA = torch.cuda.is_available()
num_classes = 80
colors = pkl.load(open("pallete", "rb"))
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")
model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32
if CUDA:
    model.cuda()

mapx, mapy = undistortMap(frame, min_factor)
sift = cv2.xfeatures2d.SIFT_create()
count = 0
kp1 = []
lines = []
cap = cv2.VideoCapture("sample_video/" + video_file_name)
while True:
    ret, frame = cap.read()
    if ret is not True:
        break
    if count % 4 == 0:
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        foot_print, frame, bboxs = get_foot_print(gray, frame, inp_dim, confidence, num_classes, nms_thresh, CUDA, model)
        if not kp1:
            kp1, des1 = sift.detectAndCompute(gray, mask=foot_print)
        else:
            kp2, des2 = sift.detectAndCompute(gray, mask=foot_print)
            if kp2:
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                for x1, y1, x2, y2 in bboxs:
                    roi_indices = np.where(np.logical_and(np.logical_and(np.greater_equal(dst_pts.transpose()[1], y1),
                                                                         np.less_equal(dst_pts.transpose()[1], y2)),
                                                          np.logical_and(np.greater_equal(dst_pts.transpose()[0], x1),
                                                                         np.less_equal(dst_pts.transpose()[0], x2))))
                    x1s, y1s, x2s, y2s = src_pts[roi_indices[1]].transpose()[0][0], \
                                         src_pts[roi_indices[1]].transpose()[1][0], \
                                         dst_pts[roi_indices[1]].transpose()[0][0], \
                                         dst_pts[roi_indices[1]].transpose()[1][0]
                    mags, angles = cv2.cartToPolar((x1s - x2s), (y1s - y2s), angleInDegrees=True)
                    if angles is not None:
                        if len(angles) > 10:
                            lower_angle_thresh, upper_angle_thresh, lower_mag_thresh, upper_mag_thresh = get_thresholds(
                                angles, mags)
                            for x1, y1, x2, y2, mag, angle in zip(x1s, y1s, x2s, y2s, mags, angles):
                                if mag >= 20 and upper_mag_thresh >= mag >= lower_mag_thresh \
                                        and upper_angle_thresh >= angle >= lower_angle_thresh:
                                    cv2.circle(frame, (int(x1), int(y1)), 3, (255, 0, 0), -1)
                                    cv2.circle(frame, (int(x2), int(y2)), 3, (0, 255, 0), -1)
                                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                    if y1 != y2:
                                        theta = np.arctan(-(x2 - x1) / (y2 - y1))
                                    else:
                                        if (x1 - x2) > 0:
                                            theta = -np.pi / 2
                                        else:
                                            theta = np.pi / 2
                                    rho = y1 * np.sin(theta) + x1 * np.cos(theta)
                                    lines.append([(x1, y1), (x2, y2)])
            kp1 = kp2
            des1 = des2
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 27:
                break
    count = count + 1
cap.release()
cv2.destroyAllWindows()

with open("polar/" + str(min_factor) + video_file_name, "wb") as fp:
    pkl.dump(lines, fp)
print("File has been written as polar/" + str(min_factor) + video_file_name)