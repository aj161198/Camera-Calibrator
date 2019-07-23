from calibutils import *
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2


def arg_parse():
    parser = argparse.ArgumentParser(description="Camera Calibration from traffic scene")
    parser.add_argument("--factor", dest="factor", help="Factor used for calibration.", type=int)
    parser.add_argument("--video", dest="video", help="Video to Calibrate from.")
    parser.add_argument("--height", dest="height", help="Height of camera from ground in cm.", type=int)


    return parser.parse_args()


args = arg_parse()
factor = args.factor
file_name = args.video


outfile_name = 'polar/' + str(factor) + file_name
cap = cv2.VideoCapture("sample_video/" + file_name)
ret, frame = cap.read()
mapx, mapy = undistortMap(frame, factor)
frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
h, w, c = frame.shape
P = np.eye(3, 4)

with open(outfile_name, 'rb') as fp:
    array = np.array(pkl.load(fp))

polar = transform_polar(array)
rhos, thetas, magnitude = polar[:, 0], polar[:, 1], polar[:, 2]
p, q, indices1, indices2 = get_peaks(thetas)

canvas1, canvas2 = None, None
if indices1 is not None:
    canvas1 = generate_canvas(rhos[indices1], thetas[indices1], frame)
    plt.imshow(canvas1)
    plt.show()
    plt.close("all")

if indices2 is not None:
    canvas2 = generate_canvas(rhos[indices2], thetas[indices2], frame)
    plt.imshow(canvas2)
    plt.show()
    plt.close("all")

if canvas1 is not None and canvas2 is not None:
    (u1, v1), (u2, v2) = get_vanishing_points(canvas1, canvas2)
    cx, cy = w / 2, h / 2
    f_ = ((u1 - cx) * (cx - u2) + (v1 - cy) * (cy - v2))
    if f_ > 0:
        f = ((u1 - cx) * (cx - u2) + (v1 - cy) * (cy - v2)) ** 0.5
        K = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
        H = int(args.height)
        P = computeProjection(H, K, u1, v1, u2, v2)
        print("The computed Projection Matrix is : - ")
        print(P)
        orthoProjection(P, K, frame)
