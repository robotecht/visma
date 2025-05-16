#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot

def main():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Get intrinsics once
    profile = pipeline.get_active_profile()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                              [0, intr.fy, intr.ppy],
                              [0, 0, 1]])
    dist_coeffs = np.array([0, 0, 0, 0, 0])  # RealSense has negligible distortion

    print("Camera matrix:\n", camera_matrix)

    # ArUco setup
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    marker_length = 0.03  # meters

    try:
        while True:
            # Wait for a coherent color frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                ids = ids.flatten()
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

                for i, marker_id in enumerate(ids):
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], marker_length, camera_matrix, dist_coeffs)

                    rvec = rvecs[0][0]
                    tvec = tvecs[0][0]

                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5)

                    # Convert rotation vector to quaternion
                    R_mat, _ = cv2.Rodrigues(rvec)
                    quat = SciRot.from_matrix(R_mat).as_quat()  # x, y, z, w

                    print(f"\nDetected Marker ID: {marker_id}")
                    print(f"Translation (x, y, z) [meters]: {tvec}")
                    print(f"Quaternion (x, y, z, w): {quat}")

            cv2.imshow('RealSense ArUco Detection', color_image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
