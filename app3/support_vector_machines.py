import os
import numpy as np
import plotly.graph_objs as go
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


def plot_decision_boundary(X, y, clf, show_support_vectors=True):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure(data=[
        go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z, 
            contours=dict(
                type='constraint',
                operation='=',
                value=0,
            ),
            line_width=3,
            line=dict(
                color='black',
                dash=None
            ),
            colorscale='RdBu',
            showlegend=False
        ),
        go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=Z, 
            contours=dict(
                type='constraint',
                operation='][',
                value=[-1, 1],
                coloring='lines'
            ),
            line_width=2,
            line=dict(
                color='grey',
                dash='dash'
            ),
            colorscale='RdBu',
            showlegend=False
        ),
        go.Scatter(
            x=X[:, 0], y=X[:, 1], mode='markers',
            marker=dict(color=y, size=10, colorscale='RdBu', line=dict(width=2, color='Black')),
            showlegend=False
        ),
    ])
    
    if show_support_vectors:
        fig.add_trace(go.Scatter(
            x=clf.support_vectors_[:, 0], y=clf.support_vectors_[:, 1], mode='markers',
            marker=dict(color='White', size=14, line=dict(width=2, color='Black')),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=clf.support_vectors_[:, 0], y=clf.support_vectors_[:, 1], mode='markers',
            marker=dict(color=y, size=10, colorscale='RdBu'),
            showlegend=False
        ))

    fig.update_layout(
        title={
            'text': f'SVM Decision Boundary with C={clf.C:.1f}',
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'bottom'
        },
        margin=dict(t=30, l=40, b=40, r=40),
        xaxis_title='Feature 1',
        yaxis_title='Feature 2'
    )

    return fig


def create_app():

    X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, 
                            n_clusters_per_class=1, class_sep=2.0, random_state=42)

    app = Dash(__name__, routes_pathname_prefix='/support-vector-machines/')

    app.layout = html.Div(
        style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'max-width': '1200px', 'margin': '0 auto', 'font-family': 'Arial, sans-serif'}, children=[

            dcc.Graph(id='C-graph', style={'height': '450px', 'width': '100%'}),

            html.Div(style={'display': 'flex', 'justify-content': 'space-around', 'margin-bottom': '10px'}, children=[
                html.Div(style={'border': '2px solid #ddd', 'padding': '10px', 'border-radius': '10px', 'width': '45%'}, children=[
                    html.H4('Parameter C', style={'text-align': 'center', 'font-size': '16px', 'margin-top': '0px', 'margin-bottom': '0px'}),
                    dcc.Slider(id='C-slider', min=0.1, max=10.0, step=0.1, value=1.0, marks={i: str(i) for i in range(0, 11)}, tooltip={"placement": "bottom", "always_visible": True}),
                ])
            ]),
    ])

    @app.callback(
        Output('C-graph', 'figure'),
        [Input('C-slider', 'value')])

    def update_graph_C(C_value):
        svm = SVC(kernel='linear', C=C_value)
        svm.fit(X, y)
        return plot_decision_boundary(X, y, clf=svm, show_support_vectors=True)
    
    return app