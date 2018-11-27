# Dash GUI for cryptocurrency trade bot. This interface shows some of the technical indicators used to generate trading signals.
# Gary Pate 2018 -- Still a work in progress with much work to be done on the signal generation
# Currently connects to a Mongo DB server where indicator data is stored from exchange API updates

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools
import numpy as np
from pymongo import MongoClient
from pymongo import errors as pe
from pymongo import ReadPreference
from sklearn.metrics import roc_curve, precision_recall_curve
import pandas as pd
from mdb_config import mdb_key

app = dash.Dash('Bittrex Trading Bot UI')
app.config['SERVER_NAME'] = 'acbotgo.com'

ORDER_MARKETS = ['BTC-LTC', 'BTC-ETH',  'BTC-XMR', 'BTC-NEO', 'BTC-XLM', 'BTC-POWR', 'BTC-ADA', 'BTC-KMD', 'BTC-PIVX', 'BTC-OMG', 'BTC-LSK']

TRAIN_VERSION = '2411_dem_1'

port = 7100


def connect_mongo():
    try:
        connection = MongoClient(mdb_key,
            maxPoolSize=5, connect=True,
            read_preference=ReadPreference.NEAREST,
            readPreference='secondaryPreferred')

        return connection

    except pe.ServerSelectionTimeoutError as e:
        return 'connection fail'

def retrieve_test_from_mongo(market, version, db):

    collection = db.testbed['test_' + version]
    data = collection.find_one({'market': market})
    pred = np.array(data['pred'])
    labels = np.array(data['labels'])
    close = np.array(data['close'])
    date = np.array(data['date'])
    null1 = np.array(data['null1'])
    null2= np.array(data['null2'])
    prob1 = np.array(data['prob1'])
    prob2 = np.array(data['prob2'])

    ls = []
    for n in range(1, 8):
        ls.append(np.array(data['ft{}'.format(n)]))

    return pred, labels, close, date, null1, null2, prob1, prob2, np.column_stack(ls)

def retrieve_train_from_mongo(market, version, db):

    collection = db.testbed['train_' + version]
    data = collection.find_one({'market': market})
    labels = np.array(data['labels'])
    close = np.array(data['close'])
    date = np.array(data['date'])
    loss1 = np.array(data['loss1'])
    loss2 = np.array(data['loss2'])
    val_loss1 = np.array(data['val_loss1'])
    val_loss2 = np.array(data['val_loss2'])
    acc1 = np.array(data['acc1'])
    acc2 = np.array(data['acc2'])
    val_acc1 = np.array(data['val_acc1'])
    val_acc2 = np.array(data['val_acc2'])


    return labels, close, date, loss1, loss2, val_loss1, val_loss2, acc1, acc2, val_acc1, val_acc2

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

def correlate(npa):
    corr_df_ = pd.DataFrame(npa)
    return corr_df_.corr()

def text_box(market, version):
    db = connect_mongo()
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
        ]
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

    db = connect_mongo()

    _, y_true, close, date, null1, null2, prob1, prob2, feat_arr = retrieve_test_from_mongo(market, version=TRAIN_VERSION, db=db)
    labels, close_train, date_train, loss1, loss2, val_loss1, val_loss2, acc1, acc2, val_acc1, val_acc2 = retrieve_train_from_mongo(market, version=TRAIN_VERSION, db=db)

    db.close()

    pred = argmax(null1, null2, prob1, prob2)

    labels_1 = np.zeros_like(y_true)
    labels_2 = np.zeros_like(y_true)

    labels_1[y_true == 1] = 1
    labels_2[y_true == 2] = 1

    roc_1 = roc_curve(labels_1[-prob1.shape[0]:], prob1)
    roc_2 = roc_curve(labels_2[-prob1.shape[0]:], prob2)

    corr_df = correlate(feat_arr)

    ht_labels = ['CCA', 'Bollinger', 'Difference', 'OB Volume', 'Kurtosis', 'Skew', 'RSI']
    trace_heat0 = go.Heatmap(z=corr_df.values.tolist(), colorscale='Cividis',
                             x=ht_labels, y=ht_labels,
                             colorbar=dict(len=0.3, y=0.125))

    trace_roc1 = go.Scatter(
        x=roc_1[0],
        y=roc_1[1],
        line=dict(color='rgb(0,220,0)', width=1.5),
        showlegend=False
    )

    trace_roc2 = go.Scatter(
        x=roc_2[0],
        y=roc_2[1],
        line=dict(color='rgb(220,0,0)', width=1.5),
        showlegend=False
    )

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

    tit1 = 'Prediction on unseen price data - {}'.format(market)
    tit2 = 'Probability of Buy label (green) and Null label (grey)'
    tit3 = 'Probability of Sell label (red) and Null label (grey)'
    tit4 = 'Training set showing Buy (green) and Sell (green) labels'
    tit5 = 'Receiver Operating Characteristic'
    tit6 = 'Train Acc. Buy - Train (Red) Val. (Blue)'
    tit7 = 'Train Acc. Sell - Train (Red) Val. (Blue)'
    tit8 = 'Feature Correlation Heatmap'

    fig_ta = tools.make_subplots(rows=6,
                                 cols=6,
                                 specs=[[{'colspan': 6}, None, None, None, None, None],
                                        [{'colspan': 6}, None, None, None, None, None],
                                        [{'colspan': 6}, None, None, None, None, None],
                                        [{'colspan': 6}, None, None, None, None, None],
                                        [{'colspan': 2, 'rowspan': 2}, None, {'colspan': 2}, None, {'colspan': 2, 'rowspan': 2}, None],
                                        [None, None, {'colspan': 2}, None, None, None]
                                        ],
                                 horizontal_spacing=0.08,
                                 vertical_spacing=0.08,
                                 subplot_titles=(tit1, tit2, tit3, tit4, tit5, tit6, tit8, tit7))

    trace_pred = [trace_replay_0, trace_replay_kmin, trace_replay_kmax]
    trace_dec1 = [trace_null1, trace_prob1]
    trace_dec2 = [trace_null2, trace_prob2]
    trace_train = [trace_train_0, trace_train_kmin, trace_train_kmax]
    trace_roc = [trace_roc1, trace_roc2]
    trace_heat = [trace_heat0]
    trace_acc1 = [acc1, val_acc1]
    trace_acc2 = [acc2, val_acc2]

    trace_list = [trace_pred, trace_dec1, trace_dec2, trace_train, trace_roc, trace_acc1, trace_acc2, trace_heat]
    trace_positions = [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [5, 3], [6, 3], [5, 5]]

    for trace, pos in zip(trace_list, trace_positions):
        for t in trace:
            fig_ta.append_trace(t, pos[0], pos[1])

    fig_ta['layout'].update(showlegend=True,
                            height=700,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(235,230,225,255)',
                            font=dict(size=11, color='#383737'),
                            margin=go.Margin(
                                l=60,
                                r=60,
                                b=70,
                                t=20,
                                pad=2
                            ),
                    )
    fig_ta['layout']['xaxis5'].update(title='False Positive Rate')
    fig_ta['layout']['yaxis5'].update(title='True Positive Rate')
    fig_ta['layout']['xaxis8'].update(title='Training Epochs')

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
                children='A Tensorflow Deep Recurrent Neural Net that predicts entry/exit points for short term trades. Data obtained from BITTREX exchange API.'
                         'The top cell shows prediction on unseen data. The middle two cells show the decision thresholds for the class labels.'
                         'The bottom cell shows labelled data from this particular time series used to train the overall model. Different currencies can be loaded via the dropdown menu.'
                         'Model developed with Keras, Numpy and Sci-kit, visualization in Dash and hosted with Docker.',
                style={'font-family': 'helvetica', 'margin-left': 50, 'margin-right': 50, 'font-size': 16}),

            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='market_select',
                        options=drop_list(dl),
                        value=dl[0],
                    )],
                    style={'width': '20%', 'font-family': 'helvetica', 'float': 'left', 'padding': 0, 'margin-left': 80,
                           'margin-top': 10, 'margin-bottom': 10, 'font-size': 16}),

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
                ], className="twelve columns"),

                # html.Div([
                #     html.Div(id='info_box',
                #     children=text_box(dl[0], TRAIN_VERSION, v_mode))
                # ], className="two columns", style= {'padding': 0, 'margin-left': 10,
                #            'margin-right': 30, 'font-size': 8}),

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

# @app.callback(
#     dash.dependencies.Output('info_box', 'children'),
#     [dash.dependencies.Input('market_select', 'value')])
# def update_output(value):
#     return text_box(value, TRAIN_VERSION)

#if __name__ == '__main__':
#    app.run_server(debug=True, use_reloader=False, port=port)
