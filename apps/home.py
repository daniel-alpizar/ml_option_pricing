from dash import html
from dash import dcc


layout = html.Div([
    html.Div([
        html.Div([
            html.P(
                "Unveil the Secrets of Options: Explore Pricing with Interactive Data", style={"color": "#0084d6",
                                              "font-size": "23px",
                                              'margin-left': '15px',
                                              'margin-top': '15px'}
            ),
            html.P([html.P(dcc.Markdown('''''',
                                        style={"color": "#ffffff",
                                               "font-size": "17px",
                                               'margin-left': '15px',
                                               'margin-top': '15px'})),
                    html.P(dcc.Markdown(
                        '''
                        **Dive deep into the world of option pricing with our interactive dashboard!** Demystify call and put options using real-world market data from AAPL, GOOG, TSLA and SPX.

                        **Experience market dynamics like never before.** Our dynamic charts let you visualize how option prices for various expirations react to shifting market sentiment. Witness the theoretical "at-the-money" price adjust with each slide, highlighting the delicate balance between risk and reward.

                        **Unlock advanced concepts.**  Unveil the elusive implied volatility with our interactive 3D plot. See how it changes across time and price, revealing the "volatility smile" - a key concept in option pricing. This isn't just a static display; it's a dynamic canvas that lets you witness how options evolve towards expiration.

                        **Model Performance at Your Fingertips:**  Compare the accuracy of different pricing models, including traditional methods and Machine Learning techniques. Discover which models perform best under varying market conditions.                        ''',
                        style={"color": "#ffffff",
                               "font-size": "15px",
                               'margin-left': '15px',
                               'margin-right': '15px',
                               'margin-bottom': '25px',
                               'line-height': '1.2',
                               'text-align': 'justify'}
                    )),
                    ])
        ], className='home_bg eight columns')
    ], className='home_row row')
])
