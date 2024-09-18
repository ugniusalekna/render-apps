import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import base64


def create_app():
    log_dir = "/Users/ugniusalekna/Documents/NMA/render-apps/logs/09-18_23-59-22/generated_vs_real"
    image_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.png')])

    if not image_files:
        raise ValueError("No images found in the directory.")

    def encode_image(image_file):
        image_path = os.path.join(log_dir, image_file)
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"

    app = dash.Dash(__name__, 
                    external_stylesheets=[
                        "https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap",
                    ],
                    routes_pathname_prefix='/gan-progress/')
    
    app.layout = html.Div([
        
        html.H2('Image Slider', id='slider-output-container', style={'textAlign': 'center'}),
     
        html.Div([
            html.Img(id='display-image', style={
                'maxWidth': '100%',
                'height': '300px',
                'display': 'block',
                'margin': 'auto'
            }),
        ]),

        html.Div(
            dcc.Slider(
                id='epoch-slider',
                min=1,
                max=len(image_files),
                value=1,
                marks={i: str(i) for i in range(1, len(image_files) + 1)},
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            style={'width': '40%', 'margin': 'auto', 'marginTop': '50px'}
        ),
    ], style={'fontFamily': 'Open Sans'})

    @app.callback(
        [Output('display-image', 'src'),
         Output('slider-output-container', 'children')],
        [Input('epoch-slider', 'value')]
    )
    def update_image(step):
        image_file = image_files[step - 1]
        encoded_image = encode_image(image_file)
        return encoded_image, f'Images at Epoch {image_file.split("_")[-1].replace(".png", "")}'

    return app

if __name__ == "__main__":
    app = create_app()
    app.run_server(debug=True, port=8080)
