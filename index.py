from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Connect to main app.py file
from app import app

# Connect to your pages
from apps import home, ml_models, options, about

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content', children=[]),

    html.Div(
        [
            html.Div([
                html.Div([
                    html.Img(src="/assets/statistics.png", style={"width": "4.9rem"}),
                    html.H5("Options Pricing", style={'color': 'white', 'margin-top': '20px'}),
                ], className='image_title')
            ], className="sidebar-header"),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink([html.Div([
                        html.I(className="fa-solid fa-house"),
                        html.Span("Home", style={'margin-top': '3px'})], className='icon_title')],
                        href="/",
                        active="exact",
                        className="pe-3"
                    ),
                    dbc.NavLink([html.Div([
                        html.I(className="fa-solid fa-gauge"),
                        html.Span("Options Modeling", style={'margin-top': '3px'})], className='icon_title')],
                        href="/apps/options",
                        active="exact",
                        className="pe-3"
                    ),
                    dbc.NavLink([html.Div([
                        html.I(className="fa-solid fa-chart-simple"),
                        html.Span("ML Models", style={'margin-top': '3px'})], className='icon_title')],
                        href="/apps/ml_models",
                        active="exact",
                        className="pe-3"
                    ),
                    dbc.NavLink([html.Div([
                        html.I(className="fa-solid fa-circle-info"),
                        html.Span("About", style={'margin-top': '3px'})], className='icon_title')],
                        href="/apps/about",
                        active="exact",
                        className="pe-3"
                    ),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        id="bg_id",
        className="sidebar",
    )

])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/apps/ml_models':
        return ml_models.layout
    elif pathname == '/apps/options':
        return options.layout
    # elif pathname == '/apps/about':
    #     return about.layout


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
