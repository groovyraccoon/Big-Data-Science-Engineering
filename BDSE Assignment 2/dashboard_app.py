import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from data.mongo_data import df

df['review_length'] = df['review_text'].astype(str).apply(len)

app = dash.Dash(__name__)
app.title = "Hotel Review Sentiment Dashboard"

app.layout = html.Div([
    html.H1("Hotel Reviews Dashboard", style={'textAlign':'center'}),
    html.Label("Filter by Sentiment:"),
    dcc.Dropdown(
        id='sentiment_filter',
        options=[
            {'label':'All','value':'all'},
            {'label':'Positive (1)','value':1},
            {'label':'Negative (0)','value':0}
        ],
        value='all',
        clearable=False,
        style={'width':'50%'}
    ),
    html.Br(),
    dcc.Graph(id='sentiment_distribution'),
    dcc.Graph(id='review_length_distribution')
])

@app.callback(
    Output('sentiment_distribution','figure'),
    Output('review_length_distribution','figure'),
    Input('sentiment_filter','value')
)

def update_dashboard(selected_sentiment):
    filtered_df = df if selected_sentiment == 'all' else df[df['sentiment_label']==int(selected_sentiment)]
    fig1 = px.histogram(filtered_df, x='sentiment_label', color='sentiment_label',
                        title='Sentiment Distribution', labels={'sentiment_label':'Sentiment'},
                        category_orders={'sentiment_label':[0,1]}, nbins=2)
    fig2 = px.histogram(filtered_df, x='review_length', nbins=50,
                        title='Review Length Distribution')

    return fig1,fig2

if __name__ == '__main__':
    app.run(debug=True)
