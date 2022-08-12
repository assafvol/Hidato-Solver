import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from hidato_reader import *

cap = cv2.VideoCapture("images/hidato_video_4.mp4")

while True:
    ret, frame = cap.read()
    try:
        t0 = time.perf_counter()
        solution_frame = read_and_solve_hidato(frame)
        t1 = time.perf_counter()
        print(f"solved frame in {t1-t0} seconds")
    except:
        solution_frame = cv2.resize(frame, (500, 500))
    cv2.imshow('frame', frame)
    cv2.imshow('solution_frame', solution_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# for i in range(10):
#     img = cv2.imread(f"video4_frames/frame_{i}.png")
#     solved_img = read_and_solve_hidato(img, debug=True)
#     cv2.imshow(f'frame {str(i)} solution', solved_img)
#     cv2.waitKey(0)

# img = cv2.imread(f"video2_frames/frame_0.png")
# t0 = time.perf_counter()
# solved_img = read_and_solve_hidato(img, debug=True)
# print(solved_img.shape)
# t1 = time.perf_counter()
# print(f"All pipeline took {t1-t0} seconds")
# cv2.imshow(f'solution', solved_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





