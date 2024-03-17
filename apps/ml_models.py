import os
from math import ceil
import pandas as pd
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from app import app


# Construct the path to the CSV file in the assets folder
csv_file_path = os.path.join('assets', 'stacked_df.csv')

# Read the CSV file using pandas
data = pd.read_csv(csv_file_path)

# Parameters to set
ticker = 'AAPL'
option = 'C'
model1 = "Black Scholes"
model2 = "Lasso"
model3 = "Single layer"
model4 = "Deep NN"

def generate_option_scatterplot(df, model, ticker, option):
    '''Function to filter data and plot the scatterplot for the given ticker and option type'''
    
    # Set Call or Put option label
    option_label = 'C' if option == 'Call' else 'P'

    # Filtering data based on model, ticker, and option type
    option_mask = (df["Model"] == model) & (df["Ticker"] == ticker) & (df["Type"] == option_label)

    fig = go.Figure()

    # Adding the trace for the filtered data
    fig.add_trace(go.Scatter(y=df.loc[option_mask, 'EOU'], 
                             x=df.loc[option_mask, 'moneyness'], 
                             mode='markers', 
                             marker=dict(color=df.loc[option_mask, 'TTE'], 
                                         colorscale='viridis', 
                                         showscale=True, 
                                         line_width=1),
                             name=model))

    # Calculate the maximum EOU for the specified ticker
    max_eou = ceil(df.loc[df['Ticker'] == ticker, 'EOU'].max())

    # Updating layout
    fig.update_layout(
                      xaxis_title='Moneyness',
                      yaxis_title='Absolute Price Error',
                      yaxis=dict(range=[0, max_eou]),
                      paper_bgcolor='#335476',
                    #   plot_bgcolor='#335476',
                      font=dict(color="white"),
                      annotations=[
                          dict(
                              text="TTE",
                              x=1.15,
                              y=1.05,
                              showarrow=False,
                              xref="paper",
                              yref="paper",
                              align="left"
                          )
                      ],
                      width=500,
                      height=300,
                      margin=dict(t=30, l=70, r=50, b=50),
                      )
    
    fig.add_annotation(
                        text=f'{ticker} {option}s - {model} Model',
                        xref="paper", yref="paper",
                        x=0.5,
                        y=0.98,
                        showarrow=False,
                        font=dict(size=12, color='darkblue'),
                        align="center",
                        bgcolor="lightblue",
                        borderpad=4,
                        xanchor="center", yanchor="top"
                        )

    return fig



layout = html.Div([
    # Heading
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('statistics.png'),
                     style={'height': '30px'},
                     className='title_image'
                     ),
            html.H6('Option Pricing Modeling with Machine Learning',
                    style={'color': 'white'},
                    className='title'
                    ),
        ], className='logo_title'),
    ], className='title_and_drop_down_list'),

    html.Div([

        # ************ Selector Column ************
        html.Div([

            html.Div([
                html.P('Select Ticker:', className='fix_label',  style={'color': 'white'}),
                
                dcc.Dropdown(
                    id='select_ticker',
                    options=[{'label': ticker, 'value': ticker} for ticker in data['Ticker'].unique()],
                    value='AAPL',
                    className='dropdown_selector'),
            ], className='selector_column'),

            html.Div([
                html.P('Select Option :', className='fix_label',  style={'color': 'white'}),
                
                dcc.Dropdown(
                    id='select_option',
                    options=['Call', 'Put'],
                    value='Call',
                    className='dropdown_selector'),
            ], className='selector_column'),

    ], className='first_column_selector'),

        # ************ Graph Column 1 ************
        html.Div([
            dcc.Graph(id='scatterplot_chart1',
                        config={'displayModeBar': False},
                        className='scatterplot_size'),
            dcc.Graph(id='scatterplot_chart2',
                        config={'displayModeBar': False},
                        className='scatterplot_size'),
        ], className='scatterplot_column'),

        # ************ Graph Column 2 ************
        html.Div([
            dcc.Graph(id='scatterplot_chart3',
                        config={'displayModeBar': False},
                        className='scatterplot_size'),
            dcc.Graph(id='scatterplot_chart4',
                        config={'displayModeBar': False},
                        className='scatterplot_size'),
        ], className='scatterplot_column')
    ], className='create_row'),
])


@app.callback(
    [Output('scatterplot_chart1', 'figure'),
     Output('scatterplot_chart2', 'figure'),
     Output('scatterplot_chart3', 'figure'),
     Output('scatterplot_chart4', 'figure'),],
    [Input('select_ticker', 'value'), Input('select_option', 'value')]
)
def update_graph(selected_ticker, selected_option):
    figure1 = generate_option_scatterplot(data, model1, selected_ticker, selected_option)
    figure2 = generate_option_scatterplot(data, model2, selected_ticker, selected_option)  # Change as needed for a different figure
    figure3 = generate_option_scatterplot(data, model3, selected_ticker, selected_option)
    figure4 = generate_option_scatterplot(data, model4, selected_ticker, selected_option)  # Change as needed for a different figure
    
    return figure1, figure2, figure3, figure4
