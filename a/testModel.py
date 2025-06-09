"""
ROBOFRIEND – Accessible Rock‑Paper‑Scissors (MediaPipe version)
==============================================================
• מצב ידידותי: WAIT → COUNTDOWN → RESULT (Space לשיגור, Esc ליציאה).
• ספירה 3‑2‑1, חיזוי יחיד באמצעות MediaPipe‑Hands + MLP (joblib).
• לוח ניקוד על‑המסך. התשתית לחיבור Arduino נשארה כתגובות טקסט.
"""

import cv2
import numpy as np
import random
import time
import joblib
import mediapipe as mp
# import serial  # ← בטל הערה כשArduino מחובר

# ─────────────────────────────────────────────────────────────
#  טעינת מסווג ה‑landmarks ומפת תוויות
# ─────────────────────────────────────────────────────────────
clf = joblib.load("rps_landmarks.joblib")
with open("label_map.txt") as f:
    CLASS_NAMES = [ln.strip().title() for ln in f]  # ['Rock', 'Paper', 'Scissors']

# MediaPipe Hands instance
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)
draw = mp.solutions.drawing_utils

# ─────────────────────────────────────────────────────────────
#  לוגיקת משחק
# ─────────────────────────────────────────────────────────────
MOVES = ["Rock", "Paper", "Scissors"]

def decide_winner(user: str, robot: str) -> str:
    if user == robot:
        return "Draw"
    wins = {("Rock", "Scissors"), ("Paper", "Rock"), ("Scissors", "Paper")}
    return "User" if (user, robot) in wins else "Robot"

# ─────────────────────────────────────────────────────────────
#  הגדרות טיימרים וניקוד
# ─────────────────────────────────────────────────────────────
COUNTDOWN_SEC = 3        # ספירה לאחור
RESULT_SEC = 2           # הצגת תוצאה

score = {"User": 0, "Robot": 0, "Draw": 0}

STATE_WAIT, STATE_COUNT, STATE_RESULT = "WAIT", "COUNT", "RESULT"
state = STATE_WAIT
next_event = 0
last_result = ""

# ─────────────────────────────────────────────────────────────
#  לולאת מצלמה
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()

        # ניקוד בראש המסך
        cv2.putText(frame, f"User: {score['User']}  Robot: {score['Robot']}  Draw: {score['Draw']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if state == STATE_WAIT:
            cv2.putText(frame, "Press SPACE to start", (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        elif state == STATE_COUNT:
            remaining = int(next_event - now) + 1
            cv2.putText(frame, str(max(0, remaining)), (10, frame.shape[0]-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if now >= next_event:
                # חיזוי יחיד על‑פי landmark‑ים
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mp_hands.process(rgb)
                if res.multi_hand_landmarks:
                    hand = res.multi_hand_landmarks[0]
                    draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

                    vec = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark],
                                   dtype=np.float32).flatten()[None, :]
                    pred_idx = int(clf.predict(vec)[0])
                    confidence = clf.predict_proba(vec)[0][pred_idx]
                    class_name = CLASS_NAMES[pred_idx]
                else:
                    confidence = 0

                if confidence >= 0.8:
                    user_move = class_name
                else:
                    user_move = random.choice(MOVES)

                robot_move = random.choice(MOVES)
                winner = decide_winner(user_move, robot_move)
                score[winner] += 1
                last_result = f"You: {user_move} | Bot: {robot_move} → {winner}!"

                # serial_msg = f"{winner.upper()}\nHAND_{robot_move.upper()}\n"
                # SER.write(serial_msg.encode())

                state = STATE_RESULT
                next_event = now + RESULT_SEC

        elif state == STATE_RESULT:
            cv2.putText(frame, last_result, (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if now >= next_event:
                state = STATE_WAIT

        cv2.imshow("ROBOFRIEND", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32 and state == STATE_WAIT:
            state = STATE_COUNT
            next_event = now + COUNTDOWN_SEC
finally:
    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()
    # SER.close()

print("👋 Exiting.")
