import cv2

print("cv2 version:", cv2.__version__)
print("Has aruco module:", hasattr(cv2, 'aruco'))

# Try accessing an ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
print("Aruco dictionary loaded.")
