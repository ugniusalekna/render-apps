import os
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


def two_layer_nn(X, W1, b1, W2, b2, activation):

    z1 = np.dot(X, W1.T) + b1
    a1 = activation(z1)
    z2 = np.dot(a1, W2) + b2
    
    return z2


def identity(x):
    return x

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

    
def create_app():

    x_seq = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_seq = np.sin(x_seq) + np.random.normal(0, 0.25, 100).reshape(-1, 1)

    app = Dash(__name__, routes_pathname_prefix='/feedforward-neural-network/')

    app.layout = html.Div(
        style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'max-width': '800px', 'margin': '0 auto', 'font-family': 'Arial, sans-serif'}, children=[
            dcc.Graph(id='nn-graph', style={'height': '350px'}),
            
            
            html.Div(style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-bottom': '20px'}, children=[
                html.Label('Activation Function:', style={'padding': '0px', 'font-size': '16px', 'margin-right': '10px'}),
                dcc.Dropdown(
                    id='activation-func',
                    options=[
                        {'label': 'Sigmoid', 'value': 'sigmoid'},
                        {'label': 'ReLU', 'value': 'relu'},
                        {'label': 'Identity', 'value': 'identity'},
                        {'label': 'Tanh', 'value': 'tanh'}
                    ],
                    value='sigmoid',
                    style={'width': '40%', 'font-size': '16px'}
                ),
            ]),

            html.Div(style={'display': 'flex', 'justify-content': 'space-around', 'margin-bottom': '10px'}, children=[
                html.Div(style={'border': '2px solid #ddd', 'padding': '10px', 'border-radius': '10px', 'width': '45%'}, children=[
                    html.H4('Layer 1', style={'text-align': 'center', 'font-size': '16px', 'margin-top': '0px', 'margin-bottom': '0px'}),
                    html.Label('Weight 1', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='W1_0', min=-2, max=2, step=0.01, value=np.random.uniform(-2, 2), marks={-2: '-2', 0: '0', 2: '2'}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label('Weight 2', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='W1_1', min=-2, max=2, step=0.01, value=np.random.uniform(-2, 2), marks={-2: '-2', 0: '0', 2: '2'}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label('Bias 1', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='b1_0', min=-2, max=2, step=0.01, value=np.random.uniform(-2, 2), marks={-2: '-2', 0: '0', 2: '2'}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label('Bias 2', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='b1_1', min=-2, max=2, step=0.01, value=np.random.uniform(-2, 2), marks={-2: '-2', 0: '0', 2: '2'}, tooltip={"placement": "bottom", "always_visible": True}),
                ]),

                html.Div(style={'border': '2px solid #ddd', 'padding': '10px', 'border-radius': '10px', 'width': '45%'}, children=[
                    html.H4('Layer 2', style={'text-align': 'center', 'font-size': '16px', 'margin-top': '0px', 'margin-bottom': '0px'}),
                    html.Label('Weight 1', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='W2_0', min=-2, max=2, step=0.01, value=np.random.uniform(-2, 2), marks={-2: '-2', 0: '0', 2: '2'}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label('Weight 2', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='W2_1', min=-2, max=2, step=0.01, value=np.random.uniform(-2, 2), marks={-2: '-2', 0: '0', 2: '2'}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label('Bias', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='b2', min=-2, max=2, step=0.01, value=np.random.uniform(-2, 2), marks={-2: '-2', 0: '0', 2: '2'}, tooltip={"placement": "bottom", "always_visible": True}),
                ]),
        ]),

    ])

    @app.callback(
        Output('nn-graph', 'figure'),
        Input('W1_0', 'value'),
        Input('W1_1', 'value'),
        Input('b1_0', 'value'),
        Input('b1_1', 'value'),
        Input('W2_0', 'value'),
        Input('W2_1', 'value'),
        Input('b2', 'value'),
        Input('activation-func', 'value'))

    def update_graph(W1_0, W1_1, b1_0, b1_1, W2_0, W2_1, b2, activation_func):
        W1 = np.array([[W1_0], [W1_1]])
        b1 = np.array([b1_0, b1_1])
        W2 = np.array([W2_0, W2_1])
        
        activation = {'sigmoid': sigmoid, 'relu': relu, 'identity': identity, 'tanh': tanh}[activation_func]
        
        out_seq = two_layer_nn(x_seq, W1, b1, W2, b2, activation)
        
        return {
            'data': [
                go.Scatter(x=x_seq.flatten(), y=y_seq.flatten(), mode='markers', name='Data'),
                go.Scatter(x=x_seq.flatten(), y=out_seq.flatten(), mode='lines', name='Prediction', line=dict(width=3))
            ],
            'layout': go.Layout(
                title={
                    'text': f'Neural Network Predictions',
                    'x': 0.5,
                    'xanchor': 'center',
                    'y': 0.95,
                    'yanchor': 'bottom'
                },
                xaxis={'title': 'x'},
                font=dict(size=14)
            )
        }

    return app