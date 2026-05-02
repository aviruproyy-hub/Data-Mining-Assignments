import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("mnist_model.h5")

window = tk.Tk()
window.title("Digit Recognizer")

canvas = tk.Canvas(window, width=280, height=280, bg="black")
canvas.pack()

image = Image.new("L", (280, 280), 0)
draw = ImageDraw.Draw(image)

def draw_lines(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

canvas.bind("<B1-Motion>", draw_lines)

def predict():
    img = image.resize((28, 28))
    img = np.array(img) / 255.0

    img_flattened = img.reshape(1, 784)

    predictions = model.predict(img_flattened)
    prediction = np.argmax(predictions)

    result_label.config(text=f"Prediction: {prediction}")

button = tk.Button(window, text="Predict", command=predict)
button.pack()

result_label = tk.Label(window, text="Prediction: ", font=("Arial", 18))
result_label.pack()

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill=0)
    result_label.config(text="Prediction: ")

clear_button = tk.Button(window, text="Clear", command=clear)
clear_button.pack()

window.mainloop()