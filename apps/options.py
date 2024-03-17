import os
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO
import base64
# import matplotlib.pyplot as plt
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from scipy.interpolate import griddata
from app import app
from dash.exceptions import PreventUpdate
import matplotlib
matplotlib.use('Agg')  # Set the backend to TkAgg
import matplotlib.pyplot as plt

# Read the CSV file using pandas
csv_file_path = os.path.join('assets', 'df_option.csv')
df_option = pd.read_csv(csv_file_path)
df_option["Date"] = pd.to_datetime(df_option["Date"], format="%m/%d/%Y")
df_option["Expiry"] = pd.to_datetime(df_option["Expiry"], format="%m/%d/%Y")
df_option.sort_values(by=["Date", "Strike", "TTE"], inplace=True)
unique_dates = df_option['Date'].unique()

marks = {i: {'label': str(i)} for i in range(0, 251, 10)}

layout = html.Div([
    html.Div([
        # Flex container for buttons and slider
        html.Div([
            html.Button('Play', id='play-button', n_clicks=0, 
                        style={'background-color': 'gray', 'color': 'white', 'border': 'none', 'marginRight': '10px'}),
            html.Button('Stop', id='stop-button', n_clicks=0, 
                        style={'background-color': 'gray', 'color': 'white', 'border': 'none', 'marginRight': '10px'}),
            dcc.Slider(id='animation-slider', min=0, max=250, value=0, step=20, marks=marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='custom-slider'),
        ], style={'display': 'flex', 'alignItems': 'center'}), # This div groups buttons and slider in a row

        # html.Div(id='interval-counter', style={'margin-top': '20px', 'color': 'white'}),
        dcc.Interval(
            id='animation-interval',
            interval=500,  # in milliseconds
            n_intervals=0, ### Starting at 
            disabled=True          
        ),
        html.Div([
            html.Img(id='live-matplotlib-img'),
        ], style={'paddingTop': '5px','paddingBottom': '5px'}),
        html.Div([
            dcc.Graph(id='live-plotly-graph')
        ],)
    ], className='main-content-padding')
])



@app.callback(
    Output('animation-interval', 'disabled'),
    Input('play-button', 'n_clicks'),
    Input('stop-button', 'n_clicks'),
    State('animation-interval', 'disabled')
)
def toggle_animation(play_clicks, stop_clicks, is_disabled):
    # Determine which button was clicked last
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'play-button' and is_disabled:
        return False  # Enable the interval
    elif button_id == 'stop-button':
        return True  # Disable the interval

    return is_disabled


@app.callback(
    Output('interval-counter', 'children'),
    [Input('animation-slider', 'value')]
)
def update_counter(n):
    date_string = pd.to_datetime(str(unique_dates[n])).strftime('%Y-%m-%d')
    return f"Date: {date_string}"


@app.callback(
    Output('animation-slider', 'value'),
    [Input('animation-interval', 'n_intervals')]
)
def update_slider(n_intervals):
    new_value = (n_intervals * 5) % 251
    return new_value


@app.callback(Output('live-matplotlib-img', 'src'),
               [Input('animation-slider', 'value')]
             )
def update_matplotlib_img(n):
    # c_1 for Date=2024-02-23, Expiry = 2024-03-15, Type = C
    mask1 = (df_option["Date"]==unique_dates[n]) 
    mask2 = (df_option["Expiry"]==pd.to_datetime("2024-03-15", format="%Y-%m-%d"))
    mask3 = df_option["Type"]=='C'
    df_plot_strike_c_1 = df_option.loc[(mask1&mask2&mask3),"Strike"]
    df_plot_price_c_1 = df_option.loc[(mask1&mask2&mask3),"Close"]
    df_plot_underlying_c_1 = df_option.loc[(mask1&mask2&mask3),"Underlying"]

    # c_2 for Date=2024-02-23, Expiry = 2024-06-21, Type = C
    ## **** Tweaking a bit to find the date that has the option chain traded on, using 02-23
    mask1 = (df_option["Date"]==unique_dates[n]) 
    mask2 = (df_option["Expiry"]==pd.to_datetime("2024-06-21", format="%Y-%m-%d"))
    mask3 = df_option["Type"]=='C'
    df_plot_strike_c_2 = df_option.loc[(mask1&mask2&mask3),"Strike"]
    df_plot_price_c_2 = df_option.loc[(mask1&mask2&mask3),"Close"]
    df_plot_underlying_c_2 = df_option.loc[(mask1&mask2&mask3),"Underlying"]

    # c_3 for Date=2024-02-23, Expiry = 2024-09-20, Type = C
    ## **** Tweaking a bit to find the date that has the option chain traded on, using 02-23
    mask1 = (df_option["Date"]==unique_dates[n]) 
    mask2 = (df_option["Expiry"]==pd.to_datetime("2024-09-20", format="%Y-%m-%d"))
    mask3 = df_option["Type"]=='C'
    df_plot_strike_c_3 = df_option.loc[(mask1&mask2&mask3),"Strike"]
    df_plot_price_c_3 = df_option.loc[(mask1&mask2&mask3),"Close"]
    df_plot_underlying_c_3 = df_option.loc[(mask1&mask2&mask3),"Underlying"]

    # c_4 for Date=2024-02-23, Expiry = 2024-12-20, Type = C
    ## **** Tweaking a bit to find the date that has the option chain traded on, using 02-23
    mask1 = (df_option["Date"]==unique_dates[n]) 
    mask2 = (df_option["Expiry"]==pd.to_datetime("2024-12-20", format="%Y-%m-%d"))
    mask3 = df_option["Type"]=='C'
    df_plot_strike_c_4 = df_option.loc[(mask1&mask2&mask3),"Strike"]
    df_plot_price_c_4 = df_option.loc[(mask1&mask2&mask3),"Close"]
    df_plot_underlying_c_4 = df_option.loc[(mask1&mask2&mask3),"Underlying"]  

    # p_1 for Date=2024-02-23, Expiry = 2024-03-15, Type = P
    mask1 = (df_option["Date"]==unique_dates[n]) 
    mask2 = (df_option["Expiry"]==pd.to_datetime("2024-03-15", format="%Y-%m-%d"))
    mask3 = df_option["Type"]=='P'
    df_plot_strike_p_1 = df_option.loc[(mask1&mask2&mask3),"Strike"]
    df_plot_price_p_1 = df_option.loc[(mask1&mask2&mask3),"Close"]
    df_plot_underlying_p_1 = df_option.loc[(mask1&mask2&mask3),"Underlying"]

    # p2 Tweaking a bit to find the date that has the option chain traded on 
    mask1 = (df_option["Date"]==unique_dates[n]) 
    mask2 = (df_option["Expiry"]==pd.to_datetime("2024-06-21", format="%Y-%m-%d"))
    mask3 = df_option["Type"]=='P'
    df_plot_strike_p_2 = df_option.loc[(mask1&mask2&mask3),"Strike"]
    df_plot_price_p_2 = df_option.loc[(mask1&mask2&mask3),"Close"]
    df_plot_underlying_p_2 = df_option.loc[(mask1&mask2&mask3),"Underlying"]

    # p3 Tweaking a bit to find the date that has the option chain traded on 
    mask1 = (df_option["Date"]==unique_dates[n]) 
    mask2 = (df_option["Expiry"]==pd.to_datetime("2024-09-20", format="%Y-%m-%d"))
    mask3 = df_option["Type"]=='P'
    df_plot_strike_p_3 = df_option.loc[(mask1&mask2&mask3),"Strike"]
    df_plot_price_p_3 = df_option.loc[(mask1&mask2&mask3),"Close"]
    df_plot_underlying_p_3 = df_option.loc[(mask1&mask2&mask3),"Underlying"]

    # p4 Tweaking a bit to find the date that has the option chain traded on 
    mask1 = (df_option["Date"]==unique_dates[n]) 
    mask2 = (df_option["Expiry"]==pd.to_datetime("2024-12-20", format="%Y-%m-%d"))
    mask3 = df_option["Type"]=='P'
    df_plot_strike_p_4 = df_option.loc[(mask1&mask2&mask3),"Strike"]
    df_plot_price_p_4 = df_option.loc[(mask1&mask2&mask3),"Close"]
    df_plot_underlying_p_4 = df_option.loc[(mask1&mask2&mask3),"Underlying"]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.3)) 
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.15)
    
    # Set the background color of the figure
    fig.patch.set_facecolor('#335476')

    # c1
    axs[0].plot(df_plot_strike_c_1, df_plot_price_c_1, marker="o", label="2024-03-15")  
    axs[0].set_xlabel('Strike', color='white', fontweight='bold')  # X-axis label
    axs[0].set_ylabel('Call Prices', color='white', fontweight='bold')  # Y-axis label
    intrinsic_value = np.maximum(df_plot_underlying_c_1 - df_plot_strike_c_1, 0)
    time_value = df_plot_price_c_1 - intrinsic_value
    for x, y, i, t in zip(df_plot_strike_c_1, df_plot_price_c_1,
                     round(intrinsic_value,2), round(time_value,2)):
        if x == 0:
            axs[0].annotate(f'Intrinsic={i}\nTimeValue={t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='left', fontsize=6,weight='bold')
        else:
            axs[0].annotate(f'{i}\n{t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontsize=6,weight='bold')

    #c2
    axs[0].plot(df_plot_strike_c_2, df_plot_price_c_2, marker="o", label="2024-06-21") 
    intrinsic_value = np.maximum(df_plot_underlying_c_2 - df_plot_strike_c_2, 0)
    time_value = df_plot_price_c_2 - intrinsic_value
    for x, y, i, t in zip(df_plot_strike_c_2, df_plot_price_c_2,
                         round(intrinsic_value,2), round(time_value,2)):
        if x == 0:
            axs[0].annotate(f'Intrinsic={i}\nTimeValue={t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='left', fontsize=6,weight='bold')
        else:
#         if x < 195:
            axs[0].annotate(f'{i}\n{t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontsize=6,weight='bold')
        
    #c3
    axs[0].plot(df_plot_strike_c_3, df_plot_price_c_3, marker="o", ) 
    intrinsic_value = np.maximum(df_plot_underlying_c_3 - df_plot_strike_c_3, 0)
    time_value = df_plot_price_c_3 - intrinsic_value
    for x, y, i, t in zip(df_plot_strike_c_3, df_plot_price_c_3,
                         round(intrinsic_value,2), round(time_value,2)):
        if x == 0:
            axs[0].annotate(f'Intrinsic={i}\nTimeValue={t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='left', fontsize=6,weight='bold')
        else:
            axs[0].annotate(f'{i}\n{t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontsize=6,weight='bold')
        
    #c4
    axs[0].plot(df_plot_strike_c_4, df_plot_price_c_4, marker="o", ) 
    intrinsic_value = np.maximum(df_plot_underlying_c_4 - df_plot_strike_c_4, 0)
    time_value = df_plot_price_c_4 - intrinsic_value
    for x, y, i, t in zip(df_plot_strike_c_4, df_plot_price_c_4,
                         round(intrinsic_value,2), round(time_value,2)):
        if x == 0:
            axs[0].annotate(f'{i} = Intrinsic Value\n{t} = Time Value', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='left', fontsize=6,weight='bold')
        else:
            axs[0].annotate(f'{i}\n{t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontsize=6,weight='bold')
            
    # x axis and background for axs[0]
    axs[0].set_xlim([148,212])
    xmin, xmax = axs[0].get_xlim()
    axs[0].axvspan(xmin, df_plot_underlying_c_1.iloc[0], color='lightgreen', alpha=0.15)  
    axs[0].axvspan(df_plot_underlying_c_1.iloc[0], xmax, color='lightcoral', alpha=0.15) 
    axs[0].axvline(x=df_plot_underlying_c_1.iloc[0], color='black')
    axs[0].legend(loc='upper right')

    #p1
    axs[1].plot(df_plot_strike_p_1, df_plot_price_p_1, marker="o", )  
    intrinsic_value = np.maximum(-df_plot_underlying_p_1 + df_plot_strike_p_1, 0)
    time_value = df_plot_price_p_1 - intrinsic_value
    for x, y, i, t in zip(df_plot_strike_p_1, df_plot_price_p_1,
                         round(intrinsic_value,2), round(time_value,2)):
        if x == 0:
            axs[1].annotate(f'Intrinsic={i}\nTimeValue={t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='right', fontsize=6,weight='bold')
        else:
            axs[1].annotate(f'{i}\n{t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontsize=6,weight='bold')

    #p2
    axs[1].plot(df_plot_strike_p_2, df_plot_price_p_2, marker="o", )  
    intrinsic_value = np.maximum(-df_plot_underlying_p_2 + df_plot_strike_p_2, 0)
    time_value = df_plot_price_p_2 - intrinsic_value
    for x, y, i, t in zip(df_plot_strike_p_2, df_plot_price_p_2,
                         round(intrinsic_value,2), round(time_value,2)):
        if x == 0:
            axs[1].annotate(f'{i}=Intrinsic Value\n{t} = Time Value', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='right', fontsize=6,weight='bold')
        else:
            axs[1].annotate(f'{i}\n{t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontsize=6,weight='bold')
    
    #p3
    axs[1].plot(df_plot_strike_p_3, df_plot_price_p_3, marker="o", label="2024-09-20")  
    intrinsic_value = np.maximum(-df_plot_underlying_p_3 + df_plot_strike_p_3, 0)
    time_value = df_plot_price_p_3 - intrinsic_value
    for x, y, i, t in zip(df_plot_strike_p_3, df_plot_price_p_3,
                         round(intrinsic_value,2), round(time_value,2)):
        if x == 0:
            axs[1].annotate(f'Intrinsic={i}\nTimeValue={t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='right', fontsize=6,weight='bold')
        else:
            axs[1].annotate(f'{i}\n{t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontsize=6,weight='bold')

    #p4
    axs[1].plot(df_plot_strike_p_4, df_plot_price_p_4, marker="o", label="2024-12-20")  
    intrinsic_value = np.maximum(-df_plot_underlying_p_4 + df_plot_strike_p_4, 0)
    time_value = df_plot_price_p_4 - intrinsic_value
    for x, y, i, t in zip(df_plot_strike_p_4, df_plot_price_p_4,
                         round(intrinsic_value,2), round(time_value,2)):
        if x == 0:
            axs[1].annotate(f'Intrinsic Value = {i}\nTime Value = {t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='right', fontsize=6,weight='bold')
        else:
            axs[1].annotate(f'{i}\n{t}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontsize=6,weight='bold')
        
    # axs[1] labels and ticks
    axs[1].set_xlabel('Strike', color='white', fontweight='bold')  # X-axis label
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[0].tick_params(colors='white')  # Makes tick labels white
    axs[0].title.set_color('white')
    axs[1].tick_params(colors='white')  # Makes tick labels white
    axs[1].title.set_color('white')
    axs[1].set_ylabel('Put Prices', color='white', fontweight='bold')  # Y-axis label
    axs[1].legend(loc='upper left')

    # xaxis background for axs[1]
    axs[1].set_xlim([148,212])
    xmin, xmax = axs[1].get_xlim()
    axs[1].axvspan(xmin, df_plot_underlying_c_1.iloc[0], color='lightcoral', alpha=0.15)  
    axs[1].axvspan(df_plot_underlying_c_1.iloc[0], xmax, color='lightgreen', alpha=0.15) 
    axs[1].axvline(x=df_plot_underlying_c_1.iloc[0], color='black')

    # set y axis for both
    axs[0].set_ylim(bottom=0, top=65)
    axs[1].set_ylim(bottom=0, top=65)
    
    date_string = pd.to_datetime(str(unique_dates[n])).strftime('%Y-%m-%d')
    axs[0].set_title(f"{date_string} Call Prices", color='white', fontweight='bold', fontsize=10)
    axs[1].set_title(f"{date_string} Put Prices", color='white', fontweight='bold', fontsize=10)

    # Set the super title for the entire figure
    plt.suptitle("Call and Put Option Price Curves by Maturity Dates", fontsize=11, color='white', fontweight='bold')

    # To further customize spacing above the super title, adjust the rect parameter
    plt.subplots_adjust(top=0.85)
    
    # Convert Matplotlib figure to PNG Image
    buf = BytesIO()
    fig.savefig(buf, format='png')#, dpi=100)
    buf.seek(0)
    string = base64.b64encode(buf.read())
    plt.close(fig)
    
    src = 'data:image/png;base64,{}'.format(string.decode())
    return src


@app.callback(Output('live-plotly-graph', 'figure'),
               [Input('animation-slider', 'value')]
             )
def update_plotly_graph(n):
    mask1 = (df_option["Type"] == "P") 
    mask2 = (df_option["Date"]==unique_dates[n]) 
    df_plot_IV_surface_c = df_option.loc[mask1&mask2, ["IV","Strike","TTE"]].reset_index(drop=True)
    df_plot_IV_surface_c["TTE"] = round(df_plot_IV_surface_c["TTE"] *365,0)

    pivot_table = df_plot_IV_surface_c.pivot(index='TTE', columns='Strike', values='IV')
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values

    # Flatten X, Y, and Z for processing
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

    # Combine x_flat and y_flat into a single array of points
    points = np.vstack((x_flat, y_flat)).T

    # Remove NaN values from z_flat and corresponding points
    valid_points = points[~np.isnan(z_flat)]
    valid_values = z_flat[~np.isnan(z_flat)]

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(pivot_table.columns, pivot_table.index)

    # Interpolate using 'griddata'
    interpolated_z = griddata(valid_points, valid_values, (grid_x, grid_y), method='cubic')
    interpolated_z_linear = griddata(valid_points, valid_values, (grid_x, grid_y), method='linear')

    # Find NaN values after linear interpolation
    nan_mask = np.isnan(interpolated_z_linear)

    # Extrapolation (nearest) for NaN values
    interpolated_z_nearest = griddata(valid_points, valid_values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')

    # Fill in the NaN values with the nearest extrapolated values
    interpolated_z_linear[nan_mask] = interpolated_z_nearest


    ##############
    # Drawing
    fig = go.Figure(data=[go.Surface(z=interpolated_z_linear, x=X, y=Y, opacity=0.8, colorscale="Jet",
                                     cmin=0.12, cmax=0.63)])
    grid_strike = X
    grid_tte = Y
    grid_iv = interpolated_z_linear
    for j in range(grid_tte.shape[1]):  # For each column in the grid
        fig.add_trace(go.Scatter3d(x=grid_strike[:, j], y=grid_tte[:, j], z=grid_iv[:, j],
                                   mode='lines', line=dict(color='black', width=2), showlegend=False))

    for i in range(grid_strike.shape[0]):  # For each row in the grid
        fig.add_trace(go.Scatter3d(x=grid_strike[i, :], y=grid_tte[i, :], z=grid_iv[i, :],
                                   mode='lines', line=dict(color='black', width=2), showlegend=False))


    # Update plot layout
    fig.update_layout(
                    title={
                        'text': "<b>3D Surface of Call Option Prices: Strike Price, IV and Maturity</b>",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    title_font=dict(size=16, color='white', family="Arial"),
                      margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor='#335476',
                      scene=dict(zaxis = dict(dtick=0.05, title='IV', range=[0.15,0.5], titlefont=dict(color='white'), tickfont=dict(color='white')),
                                 yaxis = dict(dtick=60, title='TTE (Days)', range=[0,340], titlefont=dict(color='white'), tickfont=dict(color='white')),
                                 xaxis = dict(dtick=10, title='Strike', range=[150,210], titlefont=dict(color='white'), tickfont=dict(color='white')),
                                 aspectratio=dict(x=1, y=1, z=1.2)),
                      scene_camera=dict(up=dict(x=0, y=0, z=1),eye=dict(x=-1.75, y=-1.5, z=1.25)),
                    width=1000,
                    height=320 
                 )

    return fig
