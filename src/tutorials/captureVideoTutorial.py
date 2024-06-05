# import the opencv library
import cv2


vid = None
# define a video capture object
print(vid)

vid = cv2.VideoCapture(0)

print(vid)

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # print(ret)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(vid)
# After the loop release the cap object
vid.release()
print(vid)

# Destroy all the windows
cv2.destroyAllWindows()
