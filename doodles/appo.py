import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_canvas import DashCanvas

app = dash.Dash(__name__)

app.layout = html.Div([
    DashCanvas(
        id='canvas',
        width=800,
        height=200,
        lineWidth=8,
        lineColor="black",
    ),
    html.Button(id='clear', children='clear'),
])

@app.callback(
    Output('canvas', 'json_data'),
    [Input('clear', 'n_clicks')],
    [State('canvas', 'json_data')]
)
def clear_canvas(n_clicks, json_data):
    if n_clicks is None:
        return dash.no_update

    # Print json_data before clearing
    print(f"json_data before clear: {json_data}")

    # Clear the canvas
    reset_data = '{"objects":[]}'

    # Print json_data after clearing
    print(f"json_data after clear: {reset_data}")

    return reset_data

if __name__ == '__main__':
    app.run_server(debug=True, port=8081)  # Change the port here
