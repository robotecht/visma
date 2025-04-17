import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco

def main():
    # Setup RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Load ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
    parameters = aruco.DetectorParameters()

    print("📸 RealSense camera started. Press 'q' to exit.")

    try:
        while True:
            # Wait for a frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert RealSense frame to numpy image
            frame = np.asanyarray(color_frame.get_data())

            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            # Draw markers and their IDs
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                for i in range(len(ids)):
                    c = corners[i][0]
                    top_left = tuple(c[0].astype(int))
                    cv2.putText(frame, f"ID: {ids[i][0]}", top_left,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show the result
            cv2.imshow("RealSense ArUco Detection", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("🔒 RealSense stopped.")

if __name__ == "__main__":
    main()
