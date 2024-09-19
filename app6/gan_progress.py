import os
import requests
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


def fetch_github_images(repo, path):
    token = os.getenv('GITHUB_TOKEN')  
    
    if not token:
        raise ValueError("GitHub access token not found.")

    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        'User-Agent': repo,
        'Authorization': f'token {token}'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        files = response.json()
        return [file['download_url'] for file in files if file['name'].endswith('.png')]
    else:
        print(f"Error fetching files: {response.status_code}")
        print("Response body:", response.text)
        return []

    
def create_app():
    repo = "ugniusalekna/render-apps"
    path = "logs/09-18_23-00-32/generated_vs_real"

    image_urls = sorted(
        [f for f in fetch_github_images(repo, path)],
        key=lambda f: int(f.split('_')[-1].replace('.png', ''))
    )
    if not image_urls:
        raise ValueError("No images found in the directory.")
    
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
                max=len(image_urls),
                value=1,
                marks={i: str(i) for i in range(1, len(image_urls) + 1) if i % 10 == 0},
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            style={'width': '100%', 'margin': 'auto', 'marginTop': '50px'}
        ),
    ], style={'fontFamily': 'Open Sans'})

    @app.callback(
        [Output('display-image', 'src'),
         Output('slider-output-container', 'children')],
        [Input('epoch-slider', 'value')]
    )
    def update_image(step):
        image_url = image_urls[step - 1]
        return image_url, f'Real VS Synthetic Images at Epoch {image_url.split("_")[-1].replace(".png", "")}'

    return app