import cv2
import cv2.aruco as aruco

def main():
    # Load the ArUco dictionary (7x7, 1000 markers)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
    parameters = aruco.DetectorParameters()

    # Open default webcam (0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("📸 Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )

        # Draw detected markers
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Optionally: draw ID text
            for i in range(len(ids)):
                c = corners[i][0]
                top_left = tuple(c[0].astype(int))
                cv2.putText(frame, f"ID: {ids[i][0]}", top_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("ArUco Marker Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("🔒 Camera released. Exiting...")

if __name__ == "__main__":
    main()
