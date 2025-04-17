import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco

def get_realsense_pipeline():
    """Try to start RealSense pipeline. Returns None if no device is found."""
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("⚠️  No Intel RealSense device found. Falling back to webcam.")
        return None, None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline, None  # no `VideoCapture` used

def main():
    pipeline, cap = get_realsense_pipeline()

    # If no RealSense, use default camera
    if pipeline is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open any camera.")
            return

    # Load ArUco dictionary and parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
    parameters = aruco.DetectorParameters()

    print("📸 Camera started. Press 'q' to exit.")

    try:
        while True:
            # Get frame from RealSense or fallback camera
            if pipeline:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Failed to capture frame from webcam.")
                    break

            # Convert to grayscale
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

            # Display the result
            cv2.imshow("ArUco Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if pipeline:
            pipeline.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("🔒 Camera stopped.")

if __name__ == "__main__":
    main()
