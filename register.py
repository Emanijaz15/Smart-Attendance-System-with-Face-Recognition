import cv2
import os

# Ensure dataset directory exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

cam = cv2.VideoCapture(0)
cv2.namedWindow("Register Face")

name = input("Enter Name: ").strip()
if not name:
    print("Name cannot be empty!")
    exit(1)

img_counter = 0
max_images = 5

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Register Face", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = f"dataset/{name}_{img_counter+1}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1
        if img_counter >= max_images:
            print("Collected enough images.")
            break

cam.release()
cv2.destroyAllWindows()
