import math
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import plotly.graph_objects as go
from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output



def stochastic_gradient_descent(X, y, learning_rate=0.01, num_iterations=100):
    n, k = X.shape
    d = 2
    num_parameters = int(math.factorial(k + d) / (math.factorial(k) * math.factorial(d)))

    theta = np.random.randn(num_parameters, 1) # random initial_point
    path = [theta]

    X_poly = np.c_[np.ones((n, 1)), X, X**2]

    for _ in range(num_iterations):

        random_index = np.random.randint(n)
        X_i = X_poly[random_index:random_index+1]
        y_i = y[random_index:random_index+1]
        
        gradients = 2 * X_i.T.dot(X_i.dot(theta) - y_i)

        theta = theta - learning_rate * gradients
        path.append(theta)
    
    return theta, path


def plot_sgd_path_contour(path, X, y):
    theta_path = np.array(path).reshape(-1, 3)
        
    theta_1_seq = np.linspace(-30, 5, 100)
    theta_2_seq = np.linspace(-35, 10, 100)
    T1, T2 = np.meshgrid(theta_1_seq, theta_2_seq)

    X_poly = np.c_[np.ones((X.shape[0], 1)), X, X**2]
    
    Z = np.array([
        mse(y, X_poly.dot(np.array([[theta_path[-1, 0]], [t1], [t2]])))
        for t1, t2 in zip(np.ravel(T1), np.ravel(T2))
    ])
    
    Z = Z.reshape(T1.shape)
    
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        x=theta_1_seq,
        y=theta_2_seq,
        z=Z,
        colorscale='Greys',
        contours_coloring='heatmap',
        line_smoothing=0.85,
        ncontours=50,
        colorbar=dict(
            title='Empirical Risk',
            x=1.01,
            xanchor='left'
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=theta_path[:, 1],
        y=theta_path[:, 2],
        mode='markers+lines',
        name='SGD Path',
        marker=dict(color='red', size=5),
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=[theta_path[-1, 1]],
        y=[theta_path[-1, 2]],
        mode='markers',
        marker=dict(color='black', size=10, symbol='x'),
        name='Final θ'
    ))

    fig.update_layout(
        title={
            'text': "SGD Path on Empirical Risk Contour",
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'bottom'
        },
        xaxis_title='θ<sub>1</sub>',
        yaxis_title='θ<sub>2</sub>',
        xaxis=dict(
            range=[-30, 5],
            title_standoff=10,
        ),
        yaxis=dict(
            range=[-35, 10],
            title_standoff=10,
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)',
        ),
        margin=dict(l=20, r=20, t=45, b=20),
    )

    return fig


def init_data(num_samples=100):
    np.random.seed(42)

    X = 2 - 3 * np.random.normal(0, 1, num_samples)
    y = X - 2 * (X ** 2) + np.random.normal(-3, 10, num_samples)

    X = X[:, np.newaxis]

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_standardized = (X - X_mean) / X_std
    
    return X_standardized, y

    
def create_app():
    X_standardized, y = init_data()
    _, path = stochastic_gradient_descent(X_standardized, y, learning_rate=0.005, num_iterations=300)
    fig = plot_sgd_path_contour(path, X_standardized, y)
    
    app = Dash(__name__, routes_pathname_prefix='/stochastic-gradient-descent/')

    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='sgd-graph', figure=fig),
            html.Button('Refresh SGD Path', id='refresh-button', n_clicks=0,
                        style={
                            'background-color': '#f8f9fa',
                            'color': 'black',
                            'padding': '14px 20px',
                            'border': '1px solid #bbb8b8',
                            'border-radius': '5px',
                            'cursor': 'pointer',
                            'font-size': '16px',
                            'display': 'block',
                            'margin': '10px auto',
                            'width': '200px'
                        })
        ], style={'width': '80%', 'margin': '0 auto'}),
    ])

    @app.callback(
        Output('sgd-graph', 'figure'),
        Input('refresh-button', 'n_clicks'),
    )
    def update_sgd_path(n_clicks):
        if n_clicks > 0:
            _, new_path = stochastic_gradient_descent(X_standardized, y, learning_rate=0.005, num_iterations=300)
            theta_path = np.array(new_path).reshape(-1, 3)
            fig.data[1].x = theta_path[:, 1]
            fig.data[1].y = theta_path[:, 2]
            fig.data[2].x = [theta_path[-1, 1]]
            fig.data[2].y = [theta_path[-1, 2]]
            return fig
        return plot_sgd_path_contour(path, X_standardized, y)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True)
        



