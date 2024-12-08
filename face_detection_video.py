import cv2

# تحميل نموذج Haar Cascade للوجوه الأمامية والجانبية
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

# دالة لمعالجة كل إطار وتحديد الوجوه
def identify_faces_in_frame(frame):
    # تحويل الصورة إلى تدرج الرمادي
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # تحسين الصورة باستخدام مرشح غاوسي لتقليل الضوضاء
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # كشف الوجوه الأمامية
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.03, minNeighbors=4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # رسم مستطيل أزرق للوجه الأمامي

    # كشف الوجوه الجانبية
    profile_faces = profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.03, minNeighbors=4)
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # رسم مستطيل أخضر للوجه الجانبي

    # تجربة الكشف عن الوجه عند تدوير الصورة بزاوية 90 درجة
    rotated_gray = cv2.rotate(gray_frame, cv2.ROTATE_90_CLOCKWISE)
    rotated_faces = face_cascade.detectMultiScale(rotated_gray, scaleFactor=1.03, minNeighbors=4)
    for (x, y, w, h) in rotated_faces:
        cv2.rectangle(frame, (y, x), (y + h, x + w), (255, 255, 0), 2)  # مستطيل أصفر للوجوه المائلة

    return frame

# تشغيل التعرف على الفيديو
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # معالجة كل إطار
        frame = identify_faces_in_frame(frame)

        # عرض الفيديو
        cv2.imshow('Face Detection', frame)

        # اضغط على 'q' لإيقاف الفيديو
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# تشغيل البرنامج على ملف الفيديو
process_video("C:\\Users\\LENOVO\\new\\nn.mp4")
