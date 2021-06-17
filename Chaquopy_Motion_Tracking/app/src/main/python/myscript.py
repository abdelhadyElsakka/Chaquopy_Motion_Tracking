import base64
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from com.chaquo.python import Python


def convert(data):
    decoded_data = base64.b64decode(data)
    np_data = np.fromstring(decoded_data, np.uint8)
    video = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

    return video


#
# def calibrate():
#
#     # ---------------------- CALIBRATION ---------------------------
#     # termination criteria for the iterative algorithm
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     # checkerboard of size (7 x 6) is used
#     objp = np.zeros((6*7, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
#
#     # arrays to store object points and image points from all the images.
#     objpoints = []  # 3d point in real world space
#     imgpoints = []  # 2d points in image plane.
#
#     # iterating through all calibration images
#     # in the folder
#     images = glob.glob('calib_images/checkerboard/*.jpg')
#     print(images)
#     img = cv2.imread('calib_images/checkerboard/left01.jpg')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     for fname in images:
#         img = cv2.imread(fname)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # find the chess board (calibration pattern) corners
#         ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
#
#         # if calibration pattern is found, add object points,
#         # image points (after refining them)
#         if ret == True:
#             objpoints.append(objp)
#
#             # Refine the corners of the detected corners
#             corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#             imgpoints.append(corners2)
#
#             # Draw and display the corners
#             img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
#
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#     print (mtx)
#     print (dist)
#     return mtx, dist
#
#
# # c = calibrate()
c = [[[534.34144579, 0, 339.15527836],
      [0, 534.68425882, 233.84359493],
      [0, 0, 1]], [[-2.88320983e-01, 5.41079685e-02, 1.73501622e-03, -2.61333895e-04,
                    2.04110465e-01]]]


def getCords(video):
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(video)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150)
    res = []

    while (True):
        try:
            ret, frame = cap.read()

            # operations on the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # set dictionary size depending on the aruco marker selected
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

            # detector parameters can be set here (List of detection parameters[3])
            parameters = aruco.DetectorParameters_create()
            parameters.adaptiveThreshConstant = 10

            # lists of ids and the corners belonging to each id
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
                                                                  parameters=parameters)

            # font for displaying text (below)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # check if the ids list is not empty
            # if no check is added the code will crash
            if np.all(ids != None):

                # estimate pose of each marker and return the values
                # rvet and tvec-different from camera coefficients
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, c[0], c[1])

                # (rvec-tvec).any() # get rid of that nasty numpy value array error
                # print(rvec, tvec)

                for i in range(0, ids.size):
                    # draw axis for the aruco markers

                    aruco.drawAxis(frame, c[0], c[1], rvec[i], tvec[i], 0.1)
                    x = corners[0][0][0][0]
                    y = corners[0][0][0][1]

                    print("x = ", x)
                    print("y = ", y)
                    fn = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    t = (fn / fps)
                    print(fps)

                    print("t =", t)

                    res.append([x, t])
                # draw a square around the markers
                aruco.drawDetectedMarkers(frame, corners)

                # code to show ids of the marker found
                strg = ''
                for i in range(0, ids.size):
                    strg += str(ids[i][0]) + ', '

                cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                # code to show 'No Ids' when no markers are found
                cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            print('Video Ended')
            break
        # display the resulting frame
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return res


def getVelocity(d):
    varr = []
    for i in range(len(d)):
        try:
            first = d[i]
            second = d[i + 1]
            t = second[1]
            dx = second[0] - first[0]
            dt = second[1] - first[1]
            x = second[0]
            v = dx / dt
            varr.append([x, v, t])
        except:
            break

    return varr


def getAcc(varr):
    aarr = []
    xr = []
    vr = []
    ar = []
    tr = []
    for i in range(len(varr)):
        try:
            first = varr[i]
            second = varr[i + 1]
            dt = second[2] - first[2]
            dv = second[1] - first[1]
            a = dv / dt
            xr.append(second[0])
            vr.append(second[1])
            tr.append(second[2])
            ar.append(a)
            aarr.append([second[0], second[1], a, second[2]])
        except:
            break
    return aarr, xr, vr, ar, tr


def acc_filter(signal):
    order = 1

    sampling_freq = 60.08898858638626

    cutoff_freq = 6

    normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq

    numerator_coeffs, denominator_coeffs = butter(order, normalized_cutoff_freq)

    filtered_signal = lfilter(numerator_coeffs, denominator_coeffs, signal)

    return filtered_signal


def v_filter(signal):
    order = 1

    sampling_freq = 60.08898858638626

    cutoff_freq = 6

    normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq

    numerator_coeffs, denominator_coeffs = butter(order, normalized_cutoff_freq)

    filtered_signal = lfilter(numerator_coeffs, denominator_coeffs, signal)

    return filtered_signal


def main(v):
    d_list = getCords(v)
    v_list = getVelocity(d_list)
    acc_list = getAcc(v_list)

    # fig = plt.figure()
    #
    # fig(1)
    # fig(acc_list[4], acc_list[1], 'b-', label='Displacment')
    #
    # fig.canvas.draw()
    # mg=np.fromstring(fig.canvas.tostring_rgb(),dtybe=np.uint8,sep"")
    # mg=mg.reshape.(fig.canvas.get_width_height()[::-1]+(3,))
    # mg=cv2.cvtColor(mg,cv2.COLOR_RGB2BGR)
    #
    # pil_im = video.fromarray(mg)
    # buff = io.BytesIO()
    # pil_im.save(buff,format="mp4")
    # vid_str = base64.b64encode(buff.getvalue())
    #
    # return ""+str(vid_str,'utf-8')
    plt.legend()
    plt.figure(2)
    plt.plot(acc_list[4], acc_list[2], 'r-', linewidth=2, label='Velocity')
    plt.plot(acc_list[4], v_filter(acc_list[2]), 'b-', linewidth=2, label='Filterd Velocity')
    plt.legend()
    plt.figure(3)
    plt.plot(acc_list[4], acc_list[3], 'y-', linewidth=2, label='Accelertion')
    plt.plot(acc_list[4], acc_filter(acc_list[3]), 'b-', linewidth=2, label='Filterd Accelertion')
    plt.legend()
    plt.show()

    return acc_list


def make_video(data):
    # decoded_data = base64.b64decode(data)
    # np_data = np.fromstring(decoded_data, np.uint8)
    # print(np_data)
    # temp_path = tempfile.gettempdir()
    # print(temp_path)
    #
    # with open(filename, 'wb') as wfile:
    #     wfile.write(decoded_data)
    # print(filename)
    # startfile(filename)

    # cap = cv2.VideoCapture(filename)
    # success, frame = cap.read()
    # print(success)
    files_dir = str(Python.getPlatform().getApplication().getFilesDir())
    fh = open(files_dir + "/video.mp4", "wb")
    fh.write(base64.b64decode(data))
    print(fh.name)
    fh.close()
    getCords(files_dir+"video.mp4")

    # video = open("videooo.mp4", "wb")
    # video.write(decoded_data)
    # video.close()
    # cap = cv2.VideoCapture("video.mp4")
    # success, frame = cap.read()
    # print(success)

    size = 720 * 16 // 9, 720
    duration = 2
    fps = 25
    # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'X264'), fps, size)
    # for _ in range(fps * duration):
    #     out.write(np_data)
    # print(out.release())
    # cap = cv2.VideoCapture('output.avi')
    # ret, frame = cap.read()
    # print(ret,frame)
    # temp_path = tempfile.gettempdir()
    # print(temp_path)
    # with open(temp_path + '/video.mp4', 'wb') as file:
    #     file.write(decoded_string)
    # cap = cv2.VideoCapture(temp_path + '/video.mp4')
    # success, frame = cap.read()

# test('60frame.mp4')


# print(main('60frame.mp4'))
#
# def testing(str):
#     print(str)
