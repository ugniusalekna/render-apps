from PIL import Image
import base64
from io import BytesIO

import dash
from dash import dcc, html
import numpy as np
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url#, parse_jsonstring
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import json
from scipy import ndimage
from skimage import draw, morphology


def _indices_of_path(path, scale=1):
    rr, cc = [], []
    for (Q1, Q2) in zip(path[:-2], path[1:-1]):
        # int(round()) is for Python 2 compatibility
        inds = draw.bezier_curve(int(round(Q1[-1] / scale)), 
                                 int(round(Q1[-2] / scale)), 
                                 int(round(Q2[2] / scale)), 
                                 int(round(Q2[1] / scale)), 
                                 int(round(Q2[4] / scale)), 
                                 int(round(Q2[3] / scale)), 1)
        rr += list(inds[0])
        cc += list(inds[1])
    return rr, cc


def parse_jsonstring(string, shape=None, scale=1):
    if shape is None:
        shape = (500, 500)
    mask = np.zeros(shape, dtype=np.bool_)
    try:
        data = json.loads(string)
    except:
        return mask
    scale = 1
    for obj in data['objects']:
        if obj['type'] == 'image':
            scale = obj['scaleX']
        elif obj['type'] == 'path':
            scale_obj = obj['scaleX']
            inds = _indices_of_path(obj['path'], scale=scale / scale_obj)
            radius = round(obj['strokeWidth'] / 2. / scale)
            mask_tmp = np.zeros(shape, dtype=np.bool_)
            mask_tmp[inds[0], inds[1]] = 1
            mask_tmp = ndimage.binary_dilation(mask_tmp,
                                                  morphology.disk(radius))
            mask += mask_tmp
    return mask

app = dash.Dash(__name__)
server = app.server

canvas_width = 400
canvas_height = 200

app.layout = html.Div(
    [
        # Banner
        html.Div(
            [
                html.Img(src=app.get_asset_url("ocr-logo.png"), className="app__logo"),
                html.H4("Dash OCR", className="header__text"),
            ],
            className="app__header",
        ),
        # Canvas
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Draw inside the canvas with your stylus and press Predict",
                            className="section_title",
                        ),
                        html.Div(
                            DashCanvas(
                                id="canvas",
                                lineWidth=8,
                                tool="pencil",
                                width=canvas_width,
                                # height=canvas_height,
                                scale=1,
                                hide_buttons=[
                                    "zoom",
                                    "pan",
                                    "line",
                                    "pencil",
                                    "rectangle",
                                    "select",
                                ],
                                lineColor="black",
                                goButtonTitle="Predict",
                            ),
                            className="canvas-outer",
                            style={
                                "padding": "10px",
                                # "width": f"{canvas_width}px",
                                # "height": f"{canvas_height}px",
                                "border": "2px solid red",  # Added border for debugging
                                "display": "block",
                            },
                        ),
                    ],
                    className="v-card-content",
                ),
                html.Div(
                    [
                        html.B("Doodle Recognition Output", className="section_title"),
                        dcc.Loading(dcc.Markdown(id="text-output", children="")),
                    ],
                    className="v-card-content",
                    style={"margin-top": "1em"},
                ),
            ],
            className="app__content",
        ),
    ]
)


# # Callback to clear the canvas by simulating back button presses
# @app.callback(
#     Output('canvas', 'trigger'),
#     [Input('clear', 'n_clicks')],
#     [State('canvas', 'json_data')]
# )
# def clear_canvas(n_clicks_clear, json_data):
#     if n_clicks_clear is None or json_data is None:
#         return dash.no_update

#     json_data_dict = json.loads(json_data)
#     # Count the number of strokes (i.e., objects in json_data)
#     num_strokes = len(json_data_dict['objects'])

#     if num_strokes > 0:
#         return num_strokes  # Simulate pressing 'back' multiple times

#     return dash.no_update


@app.callback(
    Output("text-output", "children"),
    [Input("canvas", "json_data")],
)
def update_data(string):
    if string:
        print(f"json_data received: {string[:100]}...")  # Debugging: Print received data

        try:
            # Try parsing the json_data
            mask = parse_jsonstring(string, shape=(canvas_height, canvas_width))
            print(f"Mask shape: {mask.shape}")  # Debugging: Print mask shape
        except Exception as e:
            print(f"Error during parsing: {e}")  # Debugging: Print error details
            return "Out of Bounding Box, click clear button and try again"

        # Proceed with further processing (if needed)
        mask = (~mask.astype(bool)).astype(int)
        image_string = array_to_data_url((255 * mask).astype(np.uint8))
        img = Image.open(BytesIO(base64.b64decode(image_string[22:])))
        
        text = "Hello"  # Replace with actual text processing if needed
        print(f"Returning text: {text}")  # Debugging: Print returned text
        return text
    else:
        print("No json_data received.")  # Debugging: Print when no data is received
        raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True, port=8080)  # Change the port here
