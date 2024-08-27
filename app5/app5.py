import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_canvas
import plotly.express as px
import torch
import numpy as np
import torch.nn.functional as F
import os

from utils import select_device, load_args_and_model

device = select_device()
args, model = load_args_and_model('/Users/ugniusalekna/Documents/NMA/heroku-app/app5', device)

def preprocess_canvas(canvas_json):
    image_data = np.array(dash_canvas.parse_jsonstring(canvas_json, shape=(350, 350, 4)))
    image_data = image_data[..., 3]
    image_data = image_data.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_data).unsqueeze(0).unsqueeze(0)
    image_tensor = F.interpolate(image_tensor, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
    return image_tensor

def predict_doodle(canvas_json):
    image_tensor = preprocess_canvas(canvas_json)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        probs = probs.cpu()

    top5_probs, top5_indices = torch.topk(probs, 5)
    top5_probs = top5_probs.numpy().flatten()
    top5_indices = top5_indices.numpy().flatten()

    return top5_indices, top5_probs

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H3('Drawing Canvas', style={'textAlign': 'center'}),
        dash_canvas.DashCanvas(
            id='drawing-canvas',
            width=350,
            height=350,
            lineWidth=5,
            hide_buttons=['zoom', 'pan'],
            lineColor='black'
        )
    ], style={'display': 'inline-block', 'padding': '10px', 'vertical-align': 'top'}),

    html.Div([
        html.H3('Image for Prediction', style={'textAlign': 'center'}),
        html.Div(id='predicted-image', style={'textAlign': 'center', 'width': '350px', 'height': '350px', 'border': '1px solid black'})
    ], style={'display': 'inline-block', 'padding': '10px', 'vertical-align': 'top'}),

    html.Div([
        html.H3('Top 5 Predictions', style={'textAlign': 'center'}),
        html.Div(id='prediction-output', style={'textAlign': 'center'})
    ], style={'display': 'inline-block', 'padding': '10px', 'vertical-align': 'top'}),

    html.Div([
        html.Button('Clear Canvas', id='clear-button', style={'margin': '10px'}),
        html.Button('Predict Doodle', id='predict-button', style={'margin': '10px'})
    ], style={'textAlign': 'center', 'clear': 'both'})
])

@app.callback(
    Output('drawing-canvas', 'json_data'),
    [Input('clear-button', 'n_clicks')]
)
def clear_canvas(n_clicks):
    if n_clicks:
        return ''
    return dash.no_update

@app.callback(
    Output('predicted-image', 'children'),
    [Input('drawing-canvas', 'json_data')]
)
def update_predicted_image(canvas_json):
    if canvas_json:
        fig = px.imshow(np.array(dash_canvas.parse_jsonstring(canvas_json, shape=(350, 350, 4)))[..., 3], binary_string=True)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350, width=350)
        return dcc.Graph(figure=fig)
    return ''

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('drawing-canvas', 'json_data')]
)
def update_prediction_output(n_clicks, canvas_json):
    if n_clicks and canvas_json:
        top5_indices, top5_probs = predict_doodle(canvas_json)
        predictions = [f"{idx}: {prob*100:.1f}%" for idx, prob in zip(top5_indices, top5_probs)]
        return html.Ul([html.Li(pred) for pred in predictions])
    return ''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8058))
    app.run_server(debug=True, host='0.0.0.0', port=port)
