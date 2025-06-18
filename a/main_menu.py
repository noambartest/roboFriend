import subprocess
import sys
import ttkbootstrap as tb
from ttkbootstrap.constants import *

def launch_rps():
    subprocess.Popen([sys.executable, "RPS_Game.py"])

def launch_snake():
    subprocess.Popen([sys.executable, "snake.py"])

def main():
    root = tb.Window(themename="flatly")
    root.title("ROBOFRIEND – Main Menu")
    root.geometry("400x300")
    root.resizable(False, False)

    tb.Label(root, text="ROBOFRIEND", font=("Segoe UI", 22, "bold")).pack(pady=(30, 20))
    tb.Button(root, text="Rock-Paper-Scissors", bootstyle=PRIMARY, width=25, command=launch_rps).pack(pady=10)
    tb.Button(root, text="Snake", bootstyle=SECONDARY, width=25, command=launch_snake).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
