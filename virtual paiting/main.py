import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Canvas for drawing
canvas = None
undo_stack = []
redo_stack = []
is_drawing = False

# Default color
drawColor = (0, 0, 255)  # Red
brushThickness = 14
eraserThickness = 50

# --- Drawing helpers ---
def draw_rounded_rect(img, top_left, bottom_right, color, selected=False, radius=20):
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    img[:] = cv2.addWeighted(overlay, 1, img, 0, 0)

    # Rounded corners
    cv2.circle(img, (x1+radius, y1+radius), radius, color, -1)
    cv2.circle(img, (x2-radius, y1+radius), radius, color, -1)
    cv2.circle(img, (x1+radius, y2-radius), radius, color, -1)
    cv2.circle(img, (x2-radius, y2-radius), radius, color, -1)

    # 3D effect
    cv2.rectangle(img, (x1, y1), (x2, y2), (200,200,200), 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (50,50,50), 2)

    if selected:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 3)

def draw_eraser_icon(img, top_left, bottom_right, selected=False):
    x1, y1 = top_left
    x2, y2 = bottom_right
    w = x2 - x1
    h = y2 - y1

    # Larger pink body (back part)
    body_x1, body_y1 = x1 + 40, y1 + 20
    body_x2, body_y2 = x2 - 20, y2 - 20
    cv2.rectangle(img, (body_x1, body_y1), (body_x2, body_y2), (124, 135, 225), -1)  # punch  color fill
    cv2.rectangle(img, (body_x1, body_y1), (body_x2, body_y2), (255, 255, 255), 2)   # white outline

    # Smaller white tip (front part)
    tip_x1, tip_y1 = body_x1 - 30, body_y1 + 0
    tip_x2, tip_y2 = body_x1 + 30, body_y2 - 0
    cv2.rectangle(img, (tip_x1, tip_y1), (tip_x2, tip_y2), (255, 255, 255), -1)      # white fill
    cv2.rectangle(img, (tip_x1, tip_y1), (tip_x2, tip_y2), (100, 100, 100), 1)       # subtle outline

    # Divider line between tip and body
    cv2.line(img, (tip_x2, tip_y1), (tip_x2, tip_y2), (100, 100, 100), 2)

    # Highlight if selected
    if selected:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)



# Palette setup
blocks = [
    ((0,0),(256,125),(0,0,255)),      # Red
    ((256,0),(512,125),(0,255,255)),  # Yellow
    ((512,0),(768,125),(255,0,255)),  # Magenta
    ((768,0),(1024,125),(255,0,0)),   # Blue
]

def build_header(active_color):
    header = np.zeros((125, 1280, 3), np.uint8)
    for (tl, br, col) in blocks:
        selected = (col == active_color)
        draw_rounded_rect(header, tl, br, col, selected)
    # Eraser block
    selected = (active_color == (0,0,0))
    draw_eraser_icon(header, (1024,0), (1280,125), selected)
    return header

# Finger state helper
def fingersUp(lmList):
    fingers = []
    fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)  # Thumb
    for id in [8, 12, 16, 20]:
        fingers.append(1 if lmList[id][2] < lmList[id-2][2] else 0)
    return fingers

xp, yp = 0, 0

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    header = build_header(drawColor)
    img[0:125, 0:1280] = header
    if canvas is None:
        canvas = np.zeros_like(img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            h, w, c = img.shape
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if lmList:
                x1, y1 = lmList[8][1], lmList[8][2]  # Index fingertip
                fingers = fingersUp(lmList)

                cv2.circle(img, (x1, y1), 15, (255,255,255), cv2.FILLED)

                # Selection mode
                if fingers[1] == 1 and fingers[2] == 1:
                    xp, yp = 0, 0
                    if y1 < 125:
                        if 0 < x1 < 256: drawColor = (0,0,255)
                        elif 256 < x1 < 512: drawColor = (0,255,255)
                        elif 512 < x1 < 768: drawColor = (255,0,255)
                        elif 768 < x1 < 1024: drawColor = (255,0,0)
                        elif 1024 < x1 < 1280: drawColor = (0,0,0)

                # Drawing mode
                elif fingers[1] == 1 and fingers[2] == 0:
                    if not is_drawing:
                        undo_stack.append(canvas.copy())
                        redo_stack.clear()
                        is_drawing = True
                    cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    thickness = eraserThickness if drawColor == (0,0,0) else brushThickness
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
                    cv2.line(canvas, (xp, yp), (x1, y1), drawColor, thickness)
                    xp, yp = x1, y1
                else:
                    xp, yp = 0, 0
                    is_drawing = False

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Merge canvas
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("AirCanvas", img)
    cv2.imshow("Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(img)
        undo_stack.clear()
        redo_stack.clear()
    elif key == ord('u'):
        if undo_stack:
            redo_stack.append(canvas.copy())
            canvas = undo_stack.pop()
    elif key == ord('r'):
        if redo_stack:
            undo_stack.append(canvas.copy())
            canvas = redo_stack.pop()

cap.release()
cv2.destroyAllWindows()