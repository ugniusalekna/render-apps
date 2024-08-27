import torch
import torch.nn.functional as F
from ipycanvas import Canvas
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt

from utils import select_device, load_args_and_model
    

def clear_canvas(_):
    canvas.clear()

def start_drawing(x, y):
    global drawing
    drawing = True
    canvas.begin_path()
    canvas.move_to(x, y)

def draw(x, y):
    if drawing:
        canvas.line_to(x, y)
        canvas.stroke()

def stop_drawing(x, y):
    global drawing
    drawing = False


if __name__ == '__main__':
        
    device = select_device()
    args, model = load_args_and_model(
        # load_path="https://convolutional-neural-network.herokuapp.com/static", 
        load_path="/Users/ugniusalekna/Documents/NMA/heroku-app/app5", 
        device=device
    )
        
    canvas = Canvas(width=350, height=350, sync_image_data=True)
    canvas.stroke_style = 'black'
    canvas.line_width = 5

    drawing = False

    canvas.on_mouse_down(start_drawing)
    canvas.on_mouse_move(draw)
    canvas.on_mouse_up(stop_drawing)
    canvas.on_mouse_out(stop_drawing)

    def preprocess_canvas(canvas):
        image_data = np.array(canvas.get_image_data())
        image_data = image_data[..., 3]
        image_data = image_data.astype(np.float32) / 255.0
        image_tensor = torch.tensor(image_data).unsqueeze(0).unsqueeze(0)
        image_tensor = F.interpolate(image_tensor, size=model.image_size, mode='bilinear', align_corners=False)
        return image_tensor

    def predict_doodle(_):
        image_tensor = preprocess_canvas(canvas)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            probs = probs.cpu()

        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_probs = top5_probs.numpy()
        top5_indices = top5_indices.numpy()

        with image_output:
            clear_output(wait=True)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(image_tensor.squeeze().cpu().numpy(), cmap='gray')
            ax.axis('off')
            plt.show()

        with prediction_output:
            clear_output(wait=True)
            predictions_html = "<div style='border-radius: 10px; background-color: #f0f0f0; padding: 5px; margin: 5px; border: 1px solid #ccc;'>"
            for i in range(5):
                prob_text = f"{args.classes[top5_indices[0][i]]}: {top5_probs[0][i] * 100:.2f}%"
                predictions_html += f"<div style='margin: 2px 0; width: 250px'>{prob_text}</div>"
                predictions_html += f"<hr style='border: none; height: 2px; background-color: gray;'>"
            predictions_html += "</div>"
            
            display(widgets.HTML(predictions_html))

    clear_button = widgets.Button(
        description="Clear Canvas", 
        layout=widgets.Layout(
            border_radius='12px', 
            padding='10px 20px', 
            margin='5px', 
            height='40px', 
            align_items='center', 
            justify_content='center'
        )
    )
    predict_button = widgets.Button(
        description="Predict Doodle", 
        layout=widgets.Layout(
            border_radius='12px', 
            padding='10px 20px', 
            margin='5px', 
            height='40px', 
            align_items='center', 
            justify_content='center'
        )
    )

    clear_button.on_click(clear_canvas)
    predict_button.on_click(predict_doodle)

    button_box = widgets.HBox(
        [clear_button, predict_button], 
        layout=widgets.Layout(justify_content='center', margin='10px 0')
    )

    image_output = widgets.Output()
    prediction_output = widgets.Output()

    canvas_title = widgets.HTML("<div style='text-align: center; font-weight: bold;'>Drawing Canvas</div>")
    image_title = widgets.HTML("<div style='text-align: center; font-weight: bold;'>Image for Prediction</div>")
    prediction_title = widgets.HTML("<div style='text-align: center; font-weight: bold;'>Top 5 Predictions</div>")

    layout = widgets.HBox([
        widgets.VBox([canvas_title, canvas]),
        widgets.VBox([image_title, image_output]),
        widgets.VBox([prediction_title, prediction_output])
    ])

    display(widgets.VBox([layout, button_box], layout=widgets.Layout(align_items='center')))
