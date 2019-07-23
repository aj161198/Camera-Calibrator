import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.draw import line_aa

def undistortPoints(u_distorted, v_distorted, K):
    """
    Given a set of points from the distorted image, it converts them to undistorted points.
    R_rectilinear = Focal_length * tan(theta)  ------------------------------------------------------- i
    R_fisheye = Focal_length * theta           ------------------------------------------------------- ii
    R_rectilinear = Focal_length * tan(R_fisheye / Focal_length) ------------------------------------- iii
    Based on the equation iii , the transformation is done.

    :param u_distorted: X-coordinate of distorted points in image
    :param v_distorted: Y-coordinate of distorted points in image
    :param K: Intrinsics Matrix
    :return: undistorted coordinates
    """
    x_distorted, y_distorted = u_distorted - K[0][2], v_distorted - K[1][2]
    r_distorted = (x_distorted ** 2 + y_distorted ** 2) ** 0.5
    thetas = r_distorted / K[0][0]
    r_undistorted = K[0][0] * np.tan(thetas)

    u_undistorted, v_undistorted = np.zeros(u_distorted.shape), np.zeros(v_distorted.shape)
    indices = np.where(r_distorted != 0)
    u_undistorted[indices], v_undistorted[indices] = r_undistorted[indices] * x_distorted[indices] / r_distorted[indices] + K[0][2], r_undistorted[indices] * y_distorted[indices] / r_distorted[indices] + K[1][2]
    indices = np.where(r_distorted == 0)
    u_undistorted[indices], v_undistorted[indices] = x_distorted[indices] + K[0][2], y_distorted[indices] + K[1][2]
    return u_undistorted, v_undistorted


def undistortMap(frame, factor):
    """
    Gives the undistort map function in x and y direction. This maps can be used in remap function for the conversion
    of distorted image to undistorted image.
    the factor.
    :param frame: Frame from video.
    :param factor: Factor used for calibration
    :return: undistorted map function in x and y direction.
    """
    h,w,c = frame.shape
    u_undistorted = list(range(w)) * h
    v_undistorted = []
    for i in range(h):
        t = [i]
        y = t * w
        v_undistorted.append(y)
    v_undistorted = np.ravel(v_undistorted)
    u_undistorted = np.array(u_undistorted)
    v_undistorted = np.array(v_undistorted)
    x_undistorted, y_undistorted = u_undistorted - w / 2, v_undistorted - h / 2
    r_undistorted = (x_undistorted ** 2 + y_undistorted ** 2) ** 0.5
    diagonal = ((h ** 2) + (w ** 2)) ** 0.5

    factor = (factor + 1) / 100.
    f = diagonal / factor
    tan_theta = r_undistorted / f
    theta = np.arctan(tan_theta)
    r_distorted = f * theta
    u_distorted, v_distorted = np.zeros(u_undistorted.shape), np.zeros(v_undistorted.shape)
    indices = np.where(r_undistorted != 0)
    u_distorted[indices], v_distorted[indices] = r_distorted[indices] * x_undistorted[indices] / r_undistorted[indices] + w / 2, r_distorted[indices] * y_undistorted[indices] / r_undistorted[indices] + h / 2
    indices = np.where(r_undistorted == 0)
    u_distorted[indices], v_distorted[indices] = x_undistorted[indices] + w / 2, y_undistorted[indices] + h / 2
    mapx = np.reshape(u_distorted, (-1, w)).astype(np.float32)
    mapy = np.reshape(v_distorted, (-1, w)).astype(np.float32)

    return mapx, mapy


def get_line(xs, ys):
    """
    Given a set of x and y points it returns the best-fit line using least square approach.
    :param xs:
    :param ys:
    :return: (rho, theta) of the line as well as the error
    """
    A = np.ones((xs.shape[0], 2))
    A[:, 0], A[:, 1] = xs, ys

    B = np.ones(xs.shape[0]) * -1
    solution = np.linalg.lstsq(A, B, rcond=None)
    a, b = solution[0]
    error = solution[1][0]
    theta = np.arctan2(b, a)
    rho = -1 / ((a * a + b * b) ** 0.5)
    return rho, theta, error


def get_trajectories(video_file_name):
    """
    Uses KLTracker for getting vehicle trajectories. It then filters trajectories based on the visibility metric
    (i.e. for how many frames was a given key point is tracked), straightness metric (i.e the ratio of distance to
    displacement)
    :param video_file_name:
    :return: It return the full coordinates array with the visibility filtered indices and distance filtered indices
    """
    cap = cv2.VideoCapture(video_file_name)

    sift = cv2.xfeatures2d.SIFT_create()

    lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, frame = cap.read()
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(old_gray, None)
    old_points = (np.array(list(map(lambda p: [p.pt], kp))).astype(int)).astype(np.float32)

    color = np.random.randint(0, 255, (old_points.shape[0], 3))

    mask = np.zeros_like(frame)
    coordinates = []
    status = []
    count = 0
    while True:
        if count % 4 == 0:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_points, None, **lk_params)

            for i, (new, old) in enumerate(zip(new_points, old_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                cv2.circle(frame, (a, b), 2, color[i].tolist(), -1)
            img = cv2.add(frame, mask)
            cv2.imshow('Frame', img)

            old_gray = gray.copy()
            coordinates.append(old_points)
            old_points = new_points.reshape(-1, 1, 2)
            status.append(st)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break
            count = 0
        count = count + 1
    coordinates = np.array(coordinates)

    status = np.array(status).transpose()[0]
    nFrames = len(coordinates)

    # indices of keypoints which are tracked for more than 80% of time
    visibility_indices = np.where(np.sum(status, axis=1) >= 0.8 * np.max(np.sum(status, axis=1)))

    # indices of top 30 keypoints which shows longest displacement in the video.
    displacement_indices = np.linalg.norm(coordinates[0][visibility_indices] - coordinates[nFrames - 1][visibility_indices],axis=2).transpose()[0].argsort()[-30:][::-1]

    # displacement of top 30 keypoints
    displacement = np.linalg.norm(
        coordinates[0][visibility_indices][displacement_indices] - coordinates[nFrames - 1][visibility_indices][
            displacement_indices], axis=2)

    # distance traversed by top 30 keypoints
    distance = np.sum(np.linalg.norm(coordinates[1:, visibility_indices[0][displacement_indices]] - coordinates[:-1,
                                                                                                    visibility_indices[
                                                                                                        0][
                                                                                                        displacement_indices]],
                                     axis=3), axis=0)

    # top 10 tracks for which is (distance / displacement) ratio close to one;
    # ((distance / displacement) ratio should be close to one for nearly straight trajectories)
    ratio_indices = np.where(distance / displacement <= 1.1)[0][:10]
    keypoints_indices = displacement_indices[ratio_indices]
    return coordinates, visibility_indices, keypoints_indices


def post_process_trajectories(frame, coordinates, visibility_indices, keypoints_indices, display):
    """
    It takes the coordinates array and the filtered indices and convert it into a two (xpoints ,ypoints) 2d array which
    consists of filtered tracks across the whole video.
    :param frame:
    :param coordinates:
    :param visibility_indices:
    :param keypoints_indices:
    :param display: set it to true in order to display trajecotries.
    :return:
    """
    nFrames = len(coordinates)
    x_distorted = np.zeros((len(keypoints_indices), nFrames))
    y_distorted = np.zeros((len(keypoints_indices), nFrames))
    for i, index in zip(range(len(keypoints_indices)), keypoints_indices):
        tempx = []
        tempy = []
        color = np.random.randint(0, 255, (len(keypoints_indices), 3))
        for j in range(nFrames):
            [[x, y]] = coordinates[j][visibility_indices][index]
            cv2.circle(frame, (int(x), int(y)), 2, color[i].tolist(), -1)
            tempx.append(x)
            tempy.append(y)
        x_distorted[i, :] = tempx
        y_distorted[i, :] = tempy

    if display:
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return x_distorted, y_distorted


def computeDistortionCoefficients(K):
    """
    It gives the distortion coefficients similar to the opencv api. It only considers the barrel distortion effect.
    :param K:
    :return:
    """
    u_points = (np.random.rand(100) * K[0][2] * 2).astype(int)
    v_points = (np.random.rand(100) * K[1][2] * 2).astype(int)
    u_undistorted, v_undistorted = undistortPoints(u_points, v_points, K)
    x_distorted = u_points - K[0][2]
    y_distorted = v_points - K[1][2]
    x_undistorted = u_undistorted - K[0][2]
    y_undistorted = v_undistorted - K[1][2]
    r2 = (x_undistorted ** 2 + y_undistorted ** 2) / (K[0][0] ** 2)
    r4 = r2 ** 2
    r6 = r2 ** 3
    indices = np.where(x_undistorted != 0)
    B = x_distorted[indices] / x_undistorted[indices] - 1
    A = np.array([r2[indices], r4[indices], r6[indices]]).transpose()
    k1, k2, k3 = np.linalg.lstsq(A,B,rcond=None)[0]
    return np.array([k1, k2, 0., 0., k3])


def generate_canvas(rhos, thetas, frame):
    """
    From every line segment a line is extended in either direction. This is line is then drawn on canvas. 
    Based on the line density(i.e. number of intersections) we can identify a vanishing point
    :param rhos: 
    :param thetas: 
    :param frame: 
    :return: canvas
    """
    h, w, c = frame.shape
    canvas = np.zeros((h * 5, w * 5), np.int)
    for rho, theta in zip(rhos, thetas):
        a = np.cos(theta)
        b = np.sin(theta)
        rho = rho + (2. * h) * b + (2. * w) * a
        if b == 0:
            x1, x2 = rho / a, (rho - h * 5 * b) / a
            y1, y2 = rho, rho
        elif a == 0:
            y1, y2 = rho / b, (rho - w * 5 * a) / b
            x1, x2 = rho, rho
        else:
            x1, x2, y1, y2 = rho / a, (rho - h * 5 * b) / a, rho / b, (rho - w * 5 * a) / b

        coordinates = []
        if 0 <= x1 < w * 5:
            coordinates.append((int(x1), 0))
        if 0 <= x2 < w * 5:
            coordinates.append((int(x2), int(h * 5 - 1)))
        if 0 <= y1 < h * 5:
            coordinates.append((0, int(y1)))
        if 0 <= y2 < h * 5:
            coordinates.append((int(w * 5 - 1), (int(y2))))
        [(x1, y1), (x2, y2)] = coordinates
        rr, cc, val = line_aa(y1, x1, y2, x2)
        canvas[rr, cc] = canvas[rr, cc] + 1
    return canvas


def transform_polar(lines):
    """
    Transforms a line segments from cartesian coordinates to polar coordinates.
    :param lines: 
    :return: 
    """
    polar = []
    for [[x1, y1], [x2, y2]] in lines:

        if y1 != y2:
            theta = np.arctan(-(x2 - x1) / (y2 - y1))
        else:
            if (x1 - x2) > 0:
                theta = -np.pi / 2
            else:
                theta = np.pi / 2
        rho = ((y1 + y2) / 2) * np.sin(theta) + ((x1 + x2) / 2) * np.cos(theta)
        magnitude = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        polar.append([rho, theta, magnitude])
    return np.array(polar)


def get_peaks(thetas):
    """
    From the histogram of theta values, two clusters are selected based on the peaks
    :param thetas: 
    :return: peaks and indices of clusters
    """
    bins, z, _ = plt.hist(thetas * 180 / np.pi, bins=180, range=(-90, 90))
    data = bins.argsort()[::-1]
    indices1, indices2 = None, None
    for i in range(1, len(data)):
        if 170 <= data[0] <= 179:
            if np.absolute(data[0] - data[i]) > 25 and data[0] - 165 < data[i]:
                p = data[0] - 90
                q = data[i] - 90
                indices1a = np.where(np.absolute(thetas * 180 / np.pi - p) <= 5)
                indices1b = np.where(np.absolute(thetas[np.where(thetas >= 0)] * 180 / np.pi - p) >= 85)
                indices2 = np.where(np.absolute(thetas * 180 / np.pi - q) <= 5)
                indices1 = np.concatenate((indices1a, indices1b), axis=1)[0]
                break
        elif 0 <= data[0] <= 4:
            if np.absolute(data[0] - data[i]) > 25 and 165 - data[0] >= data[i]:
                p = data[0] - 90
                q = data[i] - 90
                indices1a = np.where(np.absolute(thetas * 180 / np.pi - p) <= 5)
                indices1b = np.where(np.absolute(thetas[np.where(thetas >= 0)] * 180 / np.pi - p) >= 85)
                indices2 = np.where(np.absolute(thetas * 180 / np.pi - q) <= 5)
                indices1 = np.concatenate((indices1a, indices1b), axis=1)[0]
                break
        else:
            if np.absolute(data[0] - data[i]) > 25:
                p = data[0] - 90
                q = data[i] - 90
                indices1 = np.where(np.absolute(thetas * 180 / np.pi - p) <= 5)
                indices2 = np.where(np.absolute(thetas * 180 / np.pi - q) <= 5)
                break
    plt.close("all")
    return p, q, indices1, indices2


def get_vanishing_points(canvas1, canvas2):
    """
    Vanishing points for each camera is calculated by finding the top 20 points with maximum intersections.
    Then the mean of such points is computed. This mean is further added with a 2 times or 3 times the standard 
    deviation. Currently 2 times of standard deviation is used, but it may vary for some videos.
    
    :param canvas1: 
    :param canvas2: 
    :return: vanishing points in x and y directions respectively.
    """
    alpha = np.max(canvas1) * 0.20
    y1, x1 = np.where(canvas1 >= np.max(canvas1) - alpha)

    alpha = np.max(canvas2) * 0.20
    y2, x2 = np.where(canvas2 >= np.max(canvas2) - alpha)
    h_, w_ = canvas1.shape[:2]
    h, w = h_ / 5, w_ / 5
    cx, cy = w / 2, h / 2
    u1m, u2m, v1m, v2m = np.sum(canvas1[y1, x1] * x1) / np.sum(canvas1[y1, x1]) - 2 * w - cx, np.sum(
        canvas2[y2, x2] * x2) / np.sum(canvas2[y2, x2]) - 2 * w - cx, np.sum(canvas1[y1, x1] * y1) / np.sum(
        canvas1[y1, x1]) - 2 * h - cy, np.sum(canvas2[y2, x2] * y2) / np.sum(canvas2[y2, x2]) - 2 * h - cy

    if abs(u1m) > abs(u2m):
        if u1m < 0:
            u1 = u1m - 2 * np.std(x1) + cx
        else:
            u1 = u1m + 2 * np.std(x1) + cx

        if canvas1[int(v1m - 2 * np.std(y1) + cy + h * 2)][int(u1 + w * 2)] > \
                canvas1[int(v1m + 2 * np.std(y1) + cy + h * 2)][int(u1 + w * 2)]:
            v1 = v1m - 2 * np.std(y1) + cy
        else:
            v1 = v1m + 2 * np.std(y1) + cy

        if v2m < 0:
            v2 = v2m - 2 * np.std(y2) + cy
        else:
            v2 = v2m + 2 * np.std(y2) + cy

        if canvas2[int(v2 + h * 2)][int(u2m + 2 * np.std(x2) + cx + w * 2)] > canvas2[int(v2 + h * 2)][
            int(u2m - 2 * np.std(x2) + cx + w * 2)]:
            u2 = u2m + 2 * np.std(x2) + cx
        else:
            u2 = u2m - 2 * np.std(x2) + cx

    else:
        if v1m < 0:
            v2 = v1m - 2 * np.std(y1) + cy
        else:
            v2 = v1m + 2 * np.std(y1) + cy
        if canvas1[int(v2 + h * 2)][int(u1m - 2 * np.std(x1) + cx + w * 2)] > canvas1[int(v2 + h * 2)][
            int(u1m + 2 * np.std(x1) + cx + w * 2)]:
            u2 = u1m - 2 * np.std(x1) + cx
        else:
            u2 = u1m + 2 * np.std(x1) + cx

        if u2m < 0:
            u1 = u2m - 2 * np.std(x2) + cx
        else:
            u1 = u2m + 2 * np.std(x2) + cx

        if canvas2[int(v2m - 2 * np.std(y2) + cy + h * 2)][int(u1 + w * 2)] > \
                canvas2[int(v2m + 2 * np.std(y2) + cy + h * 2)][int(u1 + w * 2)]:
            v1 = v2m - 2 * np.std(y2) + cy
        else:
            v1 = v2m + 2 * np.std(y2) + cy

    return (u1, v1), (u2, v2)


def computeProjection(H, K, u1, v1, u2, v2):
    """
    From the two vanishing points and the intrinsic matrix, two rotations are computed. And from these two rotations
    the third rotation is estimated. Translation estimation is done from the height of the camera from ground.
    :param H: 
    :param K: 
    :param u1: 
    :param v1: 
    :param u2: 
    :param v2: 
    :return: The final projection matrix
    """
    Kinv = np.linalg.inv(K)
    cx, cy = K[0][2], K[1][2]
    r1 = np.matmul(Kinv, np.array([[u1], [v1], [1.]])) / np.linalg.norm(np.matmul(Kinv, np.array([[u1], [v1], [1.]])))
    r2 = np.matmul(Kinv, np.array([[u2], [v2], [1.]])) / np.linalg.norm(np.matmul(Kinv, np.array([[u2], [v2], [1.]])))
    if (u1 - cx) < 0:
        r1 = -r1
    if (v2 - cy) > 0:
        r2 = -r2
    r3 = np.cross(r1.transpose(), r2.transpose()).transpose() / (np.linalg.norm(np.cross(r1.transpose(), r2.transpose()).transpose()))
    R = np.concatenate((r1,r2), axis = 1)
    R = np.concatenate((R, r3), axis = 1)
    tz = - H / R[2][2]
    T = np.array([[0.],[0.],[tz]])
    P = np.concatenate((R,T), axis = 1)
    return P


def orthoProjection(P, K, img):
    H = np.zeros((3, 3), np.float32)
    H[:, :2] = P[:, :2]
    H[:, 2] = P[:, 3]
    H = np.matmul(K, H)
    H = np.linalg.inv(H)
    M = np.float32([[1, 0, 5000], [0, 1, 500], [0, 0, 1]])
    M = np.matmul(M, H)
    dst = cv2.warpPerspective(img, M, (10000, 10000))
    plt.imshow(dst)
    plt.show()
    plt.close("all")