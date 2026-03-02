from __future__ import annotations

from pathlib import Path
from tkinter import BOTH, END, Button, Label, Listbox, Tk, filedialog, messagebox

from PIL import Image, ImageTk

from src.models.inference import predict_faces


class FaceRecognitionGUI:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Face Recognition Demo")
        self.root.geometry("900x700")

        self.label_info = Label(root, text="Choose an image and run recognition", font=("Arial", 12))
        self.label_info.pack(pady=10)

        self.btn_open = Button(root, text="Select Image", command=self.select_image)
        self.btn_open.pack(pady=5)

        self.btn_run = Button(root, text="Run Recognition", command=self.run_recognition)
        self.btn_run.pack(pady=5)

        self.image_label = Label(root)
        self.image_label.pack(fill=BOTH, expand=True, pady=10)

        self.results = Listbox(root, width=120, height=10)
        self.results.pack(pady=10)

        self.selected_path: Path | None = None

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not file_path:
            return
        self.selected_path = Path(file_path)
        self.label_info.config(text=f"Selected: {self.selected_path}")
        self._show_image(self.selected_path)

    def run_recognition(self):
        if self.selected_path is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        try:
            result = predict_faces(self.selected_path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        self.results.delete(0, END)
        self.results.insert(END, f"Faces found: {result['num_faces']}")
        for idx, pred in enumerate(result["predictions"], start=1):
            self.results.insert(
                END,
                f"{idx}. {pred['class']} | conf={pred['confidence']:.3f} | bbox={pred['bbox']}",
            )

        annotated = Path(result["annotated_image"])
        self._show_image(annotated)

    def _show_image(self, path: Path):
        image = Image.open(path)
        image.thumbnail((800, 450))
        tk_img = ImageTk.PhotoImage(image)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img


def run_gui():
    root = Tk()
    FaceRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
