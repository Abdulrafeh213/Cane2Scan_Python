import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def main():
    root = tk.Tk()
    root.title("Cane2Scan")
    root.geometry("1000x700")
    root.configure(bg="#f0f8ff")  # Light background color similar to the gradient in the image

    # Title
    title_label = tk.Label(root, text="Cane2Scan", font=("Arial", 32, "bold"), bg="#f0f8ff")
    title_label.pack(pady=20)

    # Main content frame
    content_frame = tk.Frame(root, bg="#f0f8ff")
    content_frame.pack(fill="both", expand=True, padx=20, pady=10)

    # Left section for image display
    left_frame = tk.Frame(content_frame, bg="#4682b4", width=600, height=400)
    left_frame.grid(row=0, column=0, rowspan=3, padx=10, pady=10, sticky="nsew")
    left_frame.grid_propagate(False)  # Prevent resizing

    image_label = tk.Label(left_frame, text="Image", font=("Arial", 24, "bold"), bg="#4682b4", fg="white")
    image_label.place(relx=0.5, rely=0.5, anchor="center")

    # Right section for sliders and buttons
    right_frame = tk.Frame(content_frame, bg="#f0f8ff", width=300)
    right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ns")

    # Sliders
    for _ in range(4):  # Add 4 sliders
        slider = ttk.Scale(right_frame, from_=0, to=100, orient="horizontal")
        slider.pack(pady=10, fill="x")

    # Buttons
    button_frame = tk.Frame(content_frame, bg="#f0f8ff")
    button_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ns")
    button1 = tk.Button(button_frame, text="Button", font=("Arial", 14), bg="#4682b4", fg="white")
    button1.pack(pady=5, fill="x")
    button2 = tk.Button(button_frame, text="Button", font=("Arial", 14), bg="#4682b4", fg="white")
    button2.pack(pady=5, fill="x")

    # Result section
    result_frame = tk.Frame(content_frame, bg="#4682b4", width=600, height=50)
    result_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
    result_frame.grid_propagate(False)

    result_label = tk.Label(result_frame, text="Result", font=("Arial", 24, "bold"), bg="#4682b4", fg="white")
    result_label.place(relx=0.5, rely=0.5, anchor="center")

    root.mainloop()

if __name__ == "__main__":
    main()
