import subprocess
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import RPS_Game
import snake
import tkinter as tk

def launch_rps():
    RPS_Game.build_launcher().mainloop()

def launch_snake():
    root = tk.Tk()
    root.title("Arduino Snake Game")
    snake.SnakeGame(root)
    root.mainloop()

def main():
    root = tb.Window(themename="flatly")
    root.title("ROBOFRIEND – Main Menu")
    root.geometry("400x300")
    root.resizable(False, False)

    # Title
    tb.Label(root, text="ROBOFRIEND", font=("Segoe UI", 22, "bold")).pack(pady=(30, 20))

    # Buttons
    tb.Button(root, text="Rock-Paper-Scissors", bootstyle=PRIMARY, width=25, command=launch_rps).pack(pady=10)
    tb.Button(root, text="Snake", bootstyle=SECONDARY, width=25, command=launch_snake).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
