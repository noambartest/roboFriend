from pathlib import Path
import time, random
import cv2
import numpy as np
import joblib
import mediapipe as mp
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# ────────── AI assets ──────────
clf = joblib.load("rps_landmarks.joblib")
with open("label_map.txt") as f:
    CLASS_NAMES = [ln.strip().title() for ln in f]      # ["Rock", "Paper", "Scissors"]

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.45, min_tracking_confidence=0.45,
)
draw_utils = mp.solutions.drawing_utils
MOVES = ["Rock", "Paper", "Scissors"]

# ────────── load PNG icons ──────────
ICON_DIR = Path("icons")
icons = {
    m: cv2.imread(str(ICON_DIR / f"{m.lower()}.png"), cv2.IMREAD_UNCHANGED)
    for m in MOVES
}

def overlay_icon(bgr: np.ndarray, icon: np.ndarray, x: int, y: int) -> None:
    """Draw RGBA icon onto BGR frame at (x,y), respecting alpha."""
    h, w = icon.shape[:2]
    if y + h > bgr.shape[0] or x + w > bgr.shape[1]:
        return
    roi = bgr[y:y+h, x:x+w]
    rgb, alpha = icon[:, :, :3], icon[:, :, 3] / 255.0
    inv_alpha = 1.0 - alpha
    for c in range(3):
        roi[:, :, c] = alpha * rgb[:, :, c] + inv_alpha * roi[:, :, c]

def decide_winner(user: str, bot: str) -> str:
    if user == bot:
        return "Draw"
    wins = {("Rock", "Scissors"), ("Paper", "Rock"), ("Scissors", "Paper")}
    return "User" if (user, bot) in wins else "Robot"

# ────────── OpenCV game loop ──────────
def play_game(max_rounds: int, menu_root: tb.Window) -> None:
    score = {"User": 0, "Robot": 0, "Draw": 0}
    rounds = 0
    WAIT, COUNT, RESULT, DONE = range(4)
    state, next_event = WAIT, 0
    COUNT_SEC, RESULT_SEC = 3, 2
    last_bot = last_banner = ""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        tb.messagebox.showerror("Camera error", "Webcam not found")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()

            # header scoreboard
            cv2.putText(
                frame,
                f"User {score['User']}  Bot {score['Robot']}  Draw {score['Draw']}  {rounds}/{max_rounds}",
                (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2
            )

            # states ---------------------------------------------------------
            if state == WAIT:
                cv2.putText(
                    frame, "Press SPACE", (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2
                )

            elif state == COUNT:
                cv2.putText(
                    frame, str(int(next_event - now) + 1),
                    (10, frame.shape[0]-60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3
                )
                if now >= next_event:
                    # ------------ prediction -------------
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = mp_hands.process(rgb)
                    if res.multi_hand_landmarks:
                        hand = res.multi_hand_landmarks[0]
                        draw_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                        vec = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32
                        ).flatten()[None, :]
                        idx = int(clf.predict(vec)[0])
                        conf = clf.predict_proba(vec)[0][idx]
                        user_move = CLASS_NAMES[idx] if conf >= 0.8 else random.choice(MOVES)
                    else:
                        user_move = random.choice(MOVES)

                    last_bot = random.choice(MOVES)
                    winner = decide_winner(user_move, last_bot)
                    score[winner] += 1
                    rounds += 1
                    last_banner = f"Winner: {winner.upper()}"

                    state, next_event = RESULT, now + RESULT_SEC

            elif state == RESULT:
                # banner + bot move
                cv2.putText(frame, last_banner, (10, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f"BOT: {last_bot}", (10, frame.shape[0]-30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 200, 80), 2)

                icon = icons.get(last_bot)
                if icon is not None:
                    h, w = icon.shape[:2]
                    overlay_icon(frame, icon, frame.shape[1]-w-10, 10)

                if now >= next_event:
                    state = WAIT if rounds < max_rounds else DONE

            elif state == DONE:
                cv2.putText(frame, "GAME OVER – Esc",
                            (10, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # show frame & keys
            cv2.imshow("ROBOFRIEND", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:            # Esc -> quit
                break
            if key == 32 and state == WAIT:   # Space
                state, next_event = COUNT, now + COUNT_SEC

    finally:
        cap.release()
        cv2.destroyAllWindows()
        menu_root.deiconify()

# ────────── launcher (ttkbootstrap) ──────────
def build_launcher() -> tb.Window:
    root = tb.Window(themename="flatly")
    root.title("ROBOFRIEND • Choose rounds")
    root.geometry("+600+300")
    root.resizable(False, False)

    tb.Label(root, text="Best-of rounds", font=("Segoe UI", 14, "bold")).pack(pady=(15, 5))
    rounds_var = tb.StringVar(value="3")
    tb.Combobox(root, textvariable=rounds_var,
                values=("3", "5", "10"), width=6, state="readonly").pack(pady=6)

    def start():
        root.withdraw()
        play_game(int(rounds_var.get()), root)

    tb.Button(root, text="Start Game", bootstyle=SUCCESS, command=start).pack(pady=(12, 24))
    root.bind("<Return>", lambda *_: start())
    return root

if __name__ == "__main__":
    build_launcher().mainloop()
