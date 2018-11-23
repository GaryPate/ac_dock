# Dash GUI for cryptocurrency trade bot. This interface shows some of the technical indicators used to generate trading signals.
# Gary Pate 2018 -- Still a work in progress with much work to be done on the signal generation
# Currently connects to a Mongo DB server where indicator data is stored from exchange API updates

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools
import numpy as np
from ac_dock_api import ExchangeConn
from ac_dock_config import ORDER_MARKETS, TRAIN_VERSION

app = dash.Dash('Bittrex Trading Bot UI')
app.config['SERVER_NAME'] = 'acbotgo.com'

port = 7000

def retrieve_test_from_mongo(market, version, db):

    collection = db.testbed['test_' + version]
    data = collection.find_one({'market': market})
    pred = np.array(data['pred'])
    close = np.array(data['close'])
    date = np.array(data['date'])
    null1 = np.array(data['null1'])
    null2= np.array(data['null2'])
    prob1 = np.array(data['prob1'])
    prob2 = np.array(data['prob2'])

    return pred, close, date, null1, null2, prob1, prob2

def retrieve_train_from_mongo(market, version, db):

    collection = db.testbed['train_' + version]
    data = collection.find_one({'market': market})
    labels = np.array(data['labels'])
    close = np.array(data['close'])
    date = np.array(data['date'])

    return labels, close, date

def color_selector(variable):
    ret = []
    for var in variable:
        if var == 1:
            ret.append('green')
        elif var == 2:
            ret.append('red')
        else:
            ret.append('black')
    return ret

def trend_color_sel(variable):
    ret = []
    for var in variable:
        if var == 4:
            ret.append('white')  # high
        elif var == 3:
            ret.append('yellow')  # up
        elif var == 2:
            ret.append('orange')  # low
        elif var == 1:
            ret.append('grey')  # down
        else:
            ret.append('black')
    return ret


def text_box(market, version):
    db = ExchangeConn().connect_mongo()
    con = db.testbed['test_' + version]
    data = con.find_one({'market': market})
    dash_log = np.array(data['model_summary'])
    str_log = ''.join(dash_log)
    div = html.Div(
        children=[
            dcc.Markdown('''
__Keras Model Summary__:
```
{}
```            
'''.format(str_log))
        ]# 'margin': 0
        #, className="three columns", style={'background-color': 'gray'}
    )
    return div


def argmax(null1, null2, prob1, prob2):
    y_hat = []
    for n1, n2, p1, p2 in zip(null1, null2, prob1, prob2):

        if n1 > p1 and n2 > p2:
            y_hat.append(0)
        elif p1 > n1 and n2 > p2:
            y_hat.append(1)
        elif p1 < n1 and n2 < p2:
            y_hat.append(2)
        elif p1 > p2:
            y_hat.append(1)
        elif p2 > p1:
            y_hat.append(2)
        else:
            y_hat.append(0)

    return y_hat


def plot_update_pred(market):

    db = ExchangeConn().connect_mongo()

    _, close, date, null1, null2, prob1, prob2 = retrieve_test_from_mongo(market, version=TRAIN_VERSION, db=db)
    labels, close_train, date_train = retrieve_train_from_mongo(market, version=TRAIN_VERSION, db=db)

    db.close()

    pred = argmax(null1, null2, prob1, prob2)

    trace_train_0 = go.Scatter(
        x=date_train,
        y=close_train,
        line=dict(color='rgb(125,120,100)', width=1.5),
        showlegend=False
    )

    trace_train_kmin = go.Scatter(
        x=date_train,
        y=close_train,
        mode='lines+markers',
        line=dict(color='rgb(255,255,255, 0)', width=0),
        fill='tonexty',
        marker=dict(size=15, color='green', opacity=[0.7 if x == 1 else 0 for x in labels]),
        name='Min Predict',
        showlegend=False
    )

    trace_train_kmax = go.Scatter(
        x=date_train,
        y=close_train,
        mode='lines+markers',
        line=dict(color='rgb(255,255,255, 0)', width=0),
        fill='tonexty',
        marker=dict(size=15, color='red', opacity=[0.7 if x == 2 else 0 for x in labels]),
        name='Max Predict',
        showlegend=False
    )


    trace_replay_0 = go.Scatter(
        x=date,
        y=close,
        line=dict(color='rgb(180,180,180)', width=1.5),
        showlegend=False
    )

    trace_replay_kmin = go.Scatter(
        x=date,
        y=close,
        mode='lines+markers',
        line=dict(color='rgb(255,255,255, 0)', width=0),
        fill='tonexty',
        marker=dict(size=10, color='green', opacity=[1 if x == 1 else 0 for x in pred]),
        name='Min Predict',
        showlegend=False
    )

    trace_replay_kmax = go.Scatter(
        x=date,
        y=close,
        mode='lines+markers',
        line=dict(color='rgb(255,255,255, 0)', width=0),
        fill='tonexty',
        marker=dict(size=10, color='red', opacity=[1 if x == 2 else 0 for x in pred]),
        name='Max Predict',
        showlegend=False
    )

    trace_null1 = go.Scatter(
        x=date,
        y=null1,
        line=dict(color='rgb(60,60,60)', width=1.5),
        showlegend=False
    )

    trace_null2 = go.Scatter(
        x=date,
        y=null2,
        line=dict(color='rgb(60,60,60)', width=1.5),
        showlegend=False
    )

    trace_prob1 = go.Scatter(
        x=date,
        y=prob1,
        line=dict(color='rgb(0,210,0)', width=1.5),
        showlegend=False
    )

    trace_prob2 = go.Scatter(
        x=date,
        y=prob2,
        line=dict(color='rgb(210,0,0)', width=1.5),
        showlegend=False
    )


    # Titles
    tit1 = 'Test on unseen price data - {}'.format(market)
    tit2 = 'Model Probability of Buy label (green) and Null label (grey)'
    tit3 = 'Model Probability of Sell label (red) and Null label (grey)'
    tit4 = 'Training set showing Buy (green) and Sell (green) labels'

    fig_ta = tools.make_subplots(rows=4,
                                 cols=3,
                                 specs=[[{'colspan': 3}, None, None],
                                        [{'colspan': 3}, None, None],
                                        [{'colspan': 3}, None, None],
                                        [{'colspan': 3}, None, None],
                                        ],
                                 horizontal_spacing=0.05,
                                 vertical_spacing=0.08,
                                 subplot_titles=(tit1, tit2, tit3, tit4))

    trace_pred = [trace_replay_0, trace_replay_kmin, trace_replay_kmax]
    trace_dec1 = [trace_null1, trace_prob1]
    trace_dec2 = [trace_null2, trace_prob2]
    trace_train = [trace_train_0, trace_train_kmin, trace_train_kmax]
    trace_list = [trace_pred, trace_dec1, trace_dec2, trace_train]
    trace_positions = [[1, 1], [2, 1], [3, 1], [4, 1]]

    # Places individual graphs into page based on lists
    for trace, pos in zip(trace_list, trace_positions):
        for t in trace:
            fig_ta.append_trace(t, pos[0], pos[1])

    fig_ta['layout'].update(showlegend=True,
                            height=900,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(235,230,225,255)',
                            font=dict(size=11, color='#383737'),
                            margin=go.Margin(
                                l=60,
                                r=60,
                                b=70,
                                t=20,
                                pad=2
                            )
                    )
    return fig_ta


def drop_list(dl):
    dl_construct = []
    for d in dl:
        dl_construct.append({'label': d, 'value': d})
    return dl_construct


def serve_layout():
    dl = ORDER_MARKETS

    return html.Div(
        children=[
            html.H2(
                children='Cryptocurrency Short Term Classification Model',
                style={'font-family': 'helvetica', 'margin-left': 50, 'margin-right': 50, 'font-size': 32}),

            html.H6(
                children='A Tensorflow Deep Recurrent Neural Net (LSTM) that predicts entry/exit points for short term trades. Data obtained from BITTREX exchange API.'
                         'The top cell shows prediction on unseen data. The middle two cells show the decision thresholds for the class labels.'
                         'The bottom cell shows labelled data from this particular time series used to train the overall model. Different currencies can be loaded via the dropdown menu.'
                         'Model developed with Keras, Numpy and Sci-kit, visualization in Dash and hosted with Docker.',
                style={'font-family': 'helvetica', 'margin-left': 50, 'margin-right': 50, 'font-size': 22}),

            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='market_select',
                        options=drop_list(dl),
                        value=dl[0],
                    )],
                    style={'width': '20%', 'font-family': 'helvetica', 'float': 'left', 'padding': 0, 'margin-left': 80,
                           'margin-top': 10, 'margin-bottom': 10, 'font-size': 18}),

            ], style={'height': 80, 'width': '100%', 'margin': 0, 'padding': 0}),

            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='plots_pred',
                                figure=plot_update_pred(dl[0]),
                                config={'displayModeBar': False}),
                        ], className="row", style={'border': 1, 'color': 'solid black'}),

                    ], className="row"),
                ], className="nine columns"),
                html.Div([
                    html.Div(id='info_box',
                    children=text_box(dl[0], TRAIN_VERSION))
                ], className="three columns", style= {'padding': 0, 'margin-left': 10,
                           'margin-right': 30}),
            ], className="row"),
        ]
    )


# app = dash.Dash('AutoCryptoBot Dash')
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
app.layout = serve_layout
server = app.server


@app.callback(
    dash.dependencies.Output('plots_pred', 'figure'),
    [dash.dependencies.Input('market_select', 'value')])
def update_output(value):
    return plot_update_pred(value)

@app.callback(
    dash.dependencies.Output('info_box', 'children'),
    [dash.dependencies.Input('market_select', 'value')])
def update_output(value):
    return text_box(value, TRAIN_VERSION)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, port=port)
