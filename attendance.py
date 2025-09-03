import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

def load_known_faces(dataset_dir='dataset'):
    known_encodings = []
    known_names = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = face_recognition.load_image_file(os.path.join(dataset_dir, filename))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                name = filename.split('_')[0]
                known_names.append(name)
    return known_encodings, known_names

def mark_attendance(name, csv_file='attendance.csv'):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['Name', 'Time'])
    if not ((df['Name'] == name) & (df['Time'].str[:10] == now.strftime('%Y-%m-%d'))).any():
        df = df.append({'Name': name, 'Time': dt_string}, ignore_index=True)
        df.to_csv(csv_file, index=False)
        print(f"Attendance marked for {name} at {dt_string}")
    else:
        print(f"{name} already marked present today.")

def main():
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("No registered faces found. Please register faces first.")
        return
    cam = cv2.VideoCapture(0)
    present_today = set()
    print("Press 'q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if name != "Unknown" and name not in present_today:
                mark_attendance(name)
                present_today.add(name)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
