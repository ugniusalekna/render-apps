import os
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


def function(w):
    return w**4 - 3*w**3 + 3*w

def gradient(w):
    return 4*w**3 - 9*w**2 + 3


def gradient_descent_with_momentum(func, grad, initial_point, learning_rate=0.01, momentum=0.9, 
                                   num_iterations=100, tol=1e-6):
    w = initial_point
    w_diff = 0
    path = [w]
    
    for _ in range(num_iterations):
        new_w = w - learning_rate * grad(w) + momentum * w_diff
        w_diff = new_w - w
        path.append(new_w)
        
        if abs(func(new_w) - func(w)) < tol:
            break
            
        w = new_w
    
    return np.array(path)


def plot_gradient_descent_with_momentum(initial_point, learning_rate, momentum):

    path = gradient_descent_with_momentum(function, gradient, 
                                          initial_point=initial_point, 
                                          learning_rate=learning_rate,
                                          momentum=momentum)[::-1]
    w = np.linspace(-1.5, 3, 100)
    f = function(w)

    function_trace = go.Scatter(x=w, y=f, mode='lines', name='Function')
    path_trace = go.Scatter(x=path, y=function(path), mode='lines', name='GD Path')

    arrows = [
        dict(
            x=path[i],
            y=function(path[i]),
            ax=path[i+1],
            ay=function(path[i+1]),
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red'
        ) for i in range(len(path)-1)
    ]
    
    final_point = path[0]
    final_value = function(final_point)

    final_point_annotation = dict(
        x=final_point,
        y=final_value,
        xref="x",
        yref="y",
        text=f"Final Point<br>w={final_point:.4f}<br>f(w)={final_value:.4f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="blue",
        ax=20,
        ay=-60,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        font=dict(
            family="Arial",
            size=10,
            color="black"
        )
    )
    
    arrows.append(final_point_annotation)

    layout = go.Layout(
        title={
            'text': f"Gradient Descent with Momentum (lr={learning_rate:.3f}, momentum={momentum:.2f}, start={initial_point:.3f})",
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'bottom'
        },
        margin=dict(t=30, l=40, b=40, r=40),

        xaxis={'title': 'w'},
        yaxis={'title': 'f(w)'},
        showlegend=True,
        annotations=arrows
    )

    figure = go.Figure(data=[function_trace, path_trace], layout=layout)
    return figure


def create_app():

    app = Dash(__name__, routes_pathname_prefix='/gradient_descent_with_momentum/')

    app.layout = html.Div(
        style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'max-width': '1200px', 'margin': '0 auto', 'font-family': 'Arial, sans-serif'}, children=[

            dcc.Graph(id='gd-graph', style={'height': '400px', 'width': '100%'}),

            html.Div(style={'display': 'flex', 'justify-content': 'space-around', 'margin': '0 auto'}, children=[
                html.Div(style={'border': '2px solid #ddd', 'padding': '10px', 'margin-bottom': '10px', 'border-radius': '10px', 'width': '45%'}, children=[
                    html.H4('Gradient Descent Parameters', style={'text-align': 'center', 'font-size': '16px', 'margin-top': '0px', 'margin-bottom': '0px'}),
                    html.Label('Initial Point', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='initial-point', min=-1.5, max=3, step=0.1, value=-1.10, marks={i: str(i) for i in range(-5, 6, 1)}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label('Learning Rate', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='learning-rate', min=0.001, max=0.1, step=0.001, value=0.005, marks={i/100: f"{i/100:.2f}" for i in range(2, 10, 2)}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Label('Momentum', style={'font-size': '12px', 'margin-bottom': '5px'}),
                    dcc.Slider(id='momentum', min=0.0, max=1.0, step=0.01, value=0.98, marks={i/10: f"{i/10:.1f}" for i in range(2, 10, 2)}, tooltip={"placement": "bottom", "always_visible": True}),
                ])
            ]),
    ])

    @app.callback(
        Output('gd-graph', 'figure'),
        Input('initial-point', 'value'),
        Input('learning-rate', 'value'),
        Input('momentum', 'value'))
    
    def update_graph_gd(initial_point, learning_rate, momentum):
        return plot_gradient_descent_with_momentum(initial_point, learning_rate, momentum)

    return app