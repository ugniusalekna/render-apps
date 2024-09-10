import dash
from dash import dcc, html
import dash_canvas
from dash.dependencies import Input, Output, State
import json
import numpy as np
import torch
import torch.nn.functional as F
import base64
from PIL import Image
import io

from utils import select_device, load_args_and_model


app = dash.Dash(__name__)

device = select_device()
args, model = load_args_and_model('/Users/ugniusalekna/Documents/NMA/render-apps/app5', device)

app.layout = html.Div(style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '100vh', 'flex-direction': 'column'}, children=[
    
    html.Div(style={'display': 'flex', 'justify-content': 'space-between', 'width': '80%'}, children=[
        html.Div([
            html.H2("Drawing Canvas", style={'textAlign': 'center'}),
            dash_canvas.DashCanvas(
                id='canvas',
                width=350,
                height=350,
                lineWidth=5,
                hide_buttons=['zoom', 'pan', 'reset'],
                goButtonTitle='Predict Doodle'
            )
        ], style={'flex': '1', 'margin': '20px', 'textAlign': 'center'}),
        
        html.Div([
            html.H2("Image for Prediction", style={'textAlign': 'center'}),
            html.Img(id='image-display', style={'border': '1px solid black', 'width': '300px', 'height': '300px'}),
        ], style={'flex': '1', 'margin': '20px', 'textAlign': 'center'}),
        
        html.Div([
            html.H2("Top 5 Predictions", style={'textAlign': 'center'}),
            html.Div(id='predictions', style={'fontSize': 20, 'marginTop': 20, 'border': '1px solid black', 'padding': '10px', 'minHeight': '300px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        ], style={'flex': '1', 'margin': '20px', 'textAlign': 'center'}),
    ]),

    html.Button('Predict Doodle', id='predict-button', n_clicks=0, style={'marginTop': '20px', 'padding': '10px 20px', 'fontSize': '16px'})
])

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def parse_json_data(json_data):
    data = json.loads(json_data)
    img_array = np.array(data["objects"][0]["path"], dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img

def preprocess_image(image, args):
    image = image.resize(*args.image_size).convert('L')
    image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0) / 255.0
    return image_tensor


def predict(image_tensor, model, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        probs = probs.cpu()

    top5_probs, top5_indices = torch.topk(probs, 5)
    top5_probs = top5_probs.numpy().flatten()
    top5_indices = top5_indices.numpy().flatten()

    return top5_indices, top5_probs


@app.callback(
    [Output('image-display', 'src'), Output('predictions', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('canvas', 'json_data')]
)
def update_output(n_clicks, json_data):
    if n_clicks > 0:
        print("Button clicked")
        if json_data:
            print("Canvas data received")
            try:
                # Decode the canvas data
                pil_img = parse_json_data(json_data)

                # Process the image and predict
                image_tensor = preprocess_image(pil_img, args)
                top5_indices, top5_probs = predict(image_tensor, model, device)
                
                print(f"Predictions: {top5_indices}, Probabilities: {top5_probs}")

                # Convert image to display
                img_b64 = image_to_base64(pil_img)
                img_display = f'data:image/png;base64,{img_b64}'
                
                # Create predictions output
                predictions_list = [f"Class {index}: {prob:.2f}%" for index, prob in zip(top5_indices, top5_probs)]
                predictions_output = html.Ul([html.Li(pred) for pred in predictions_list])
                
                return img_display, predictions_output
            
            except Exception as e:
                print(f"Error during processing: {e}")
                return None, f"Error during processing: {e}"

    # If no image or prediction is available, return empty values
    return "", ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
