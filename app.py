import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import re
import joblib
import numpy as np
from gensim.models import KeyedVectors
import plotly.express as px

# --- Stop words ---
stop_words = set([
    "y", "o", "el", "la", "los", "las", "de", "en", "un", "una", "es", "para", "con", "por", "no",
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[,.=:;!?"()\[\]{}\-]', ' ', text)
    tokens = re.findall(r'\b[a-záéíóúñ]+\b', text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Carga modelos TF-IDF ---
fold_tfidf = 3
dir_path_model = "model/"
model_tfidf = joblib.load(f"{dir_path_model}/fold_{fold_tfidf}/tfidf_logreg_fold_{fold_tfidf}.joblib")
vectorizer = joblib.load(f"{dir_path_model}/fold_{fold_tfidf}/tfidf_vectorizer_fold_{fold_tfidf}.joblib")
label_encoder_tfidf = joblib.load(f"{dir_path_model}/fold_{fold_tfidf}/tfidf_le_{fold_tfidf}.joblib")

# --- Carga modelos Word2Vec ---
fold_w2v = 4
model_w2v = joblib.load(f"{dir_path_model}/fold_{fold_w2v}/word2vec_logreg_fold_{fold_w2v}.joblib")
label_encoder_w2v = joblib.load(f"{dir_path_model}/fold_{fold_w2v}/word2vec_le_{fold_w2v}.joblib")
word2vec_path = "glove.6B/glove.6B.100d.word2vec.txt"
w2v_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
embedding_dim = w2v_model.vector_size

def document_vector(doc):
    words = doc.split()
    word_vecs = [w2v_model[word] for word in words if word in w2v_model]
    if len(word_vecs) == 0:
        return np.zeros(embedding_dim)
    return np.mean(word_vecs, axis=0)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

data_storage = {'df': None}

# Sidebar
sidebar = html.Div([
    html.H4("Opciones", className="text-center my-3"),
    html.Label("Selecciona modelo de predicción:"),
    dcc.Dropdown(
        id='model-selector',
        options=[
            {'label': 'TF-IDF + Regresión Logística', 'value': 'tfidf'},
            {'label': 'Word2Vec + Regresión Logística', 'value': 'w2v'}
        ],
        value='tfidf',
        clearable=False
    ),
    html.Hr(),
    html.Label("Filtrar por Level:"),
    dcc.Dropdown(id='filter-level', clearable=True, placeholder="Selecciona nivel"),
    html.Hr(),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arrastra o selecciona un archivo CSV']),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '2px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center',
            'cursor': 'pointer'
        },
        multiple=False
    ),
    html.Div(id='upload-message', className='mt-2'),
], style={
    'width': '20%',
    'height': '90vh',
    'position': 'fixed',
    'overflowY': 'auto',
    'padding': '10px',
    'borderRight': '1px solid #ddd',
})

# Main content
content = html.Div(id='main-content', style={
    'marginLeft': '21%', 'padding': '20px', 'height': '90vh', 'overflowY': 'auto'
})

app.layout = html.Div([
    dbc.Container([
        html.H1("Registros Logs", className="text-center my-4"),
    ], fluid=True),
    sidebar,
    content
])

def parse_uploaded(contents, filename, model_type):
    if contents is None:
        return None, "No hay archivo cargado"

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8'))).fillna("")
        else:
            return None, "Error: Solo archivos CSV son aceptados"
    except Exception as e:
        return None, f"Error leyendo archivo: {str(e)}"

    if 'Content' not in df.columns:
        return None, 'Error: El archivo debe contener columna "Content"'

    df['Clean_Content'] = df['Content'].apply(clean_text)

    if model_type == 'tfidf':
        X_new = vectorizer.transform(df['Clean_Content'])
        y_pred = model_tfidf.predict(X_new)
        y_pred_labels = label_encoder_tfidf.inverse_transform(y_pred)
    else:
        X_new = np.vstack(df['Clean_Content'].apply(document_vector).to_numpy())
        y_pred = model_w2v.predict(X_new)
        y_pred_labels = label_encoder_w2v.inverse_transform(y_pred)

    df['Level'] = y_pred_labels
    return df, f"Archivo {filename} cargado y procesado con modelo {model_type.upper()}."

@app.callback(
    Output("download-data", "data"),
    Input("btn-download", "n_clicks"),
    prevent_initial_call=True
)
def generate_csv(n_clicks):
    df = data_storage['df']
    if df is not None:
        return dcc.send_data_frame(df.to_csv, "registros_pred.csv", index=False)
    return None

@app.callback(
    [
        Output('upload-message', 'children'),
        Output('main-content', 'children'),
        Output('filter-level', 'options'),
        Output('filter-level', 'value'),
        Output('upload-data', 'contents')
    ],
    [
        Input('upload-data', 'contents'),
        Input('model-selector', 'value'),
        Input('filter-level', 'value'),
    ],
    [State('upload-data', 'filename')],
)
def update_data(contents, model_type, filter_level, filename):
    if contents is None and data_storage['df'] is None:
        return "", html.Div("Sube un archivo CSV para comenzar"), [], None, None

    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id in ['upload-data', 'model-selector']:
        df, message = parse_uploaded(contents, filename, model_type)
        if df is None:
            data_storage['df'] = None
            return message, html.Div(), [], None, None
        data_storage['df'] = df
        filter_value = None
    else:
        df = data_storage['df']
        message = ""
        filter_value = filter_level

    if df is None:
        return "", html.Div("Sube un archivo CSV para comenzar"), [], None, None

    dff = df[df['Level'] == filter_value].copy() if filter_value else df.copy()
    unique_levels = sorted(df['Level'].unique())
    options = [{'label': lvl, 'value': lvl} for lvl in unique_levels]

    counts = df['Level'].value_counts().to_dict()
    color_map = {
        'error': '#cd5c5c',
        'advertencia': '#ff8c00',
        'informativo': '#6495ed',
    }
    all_levels = ['Error', 'Advertencia', 'Informativo']
    counts_complete = {lvl: counts.get(lvl, 0) for lvl in all_levels}

    fig = px.bar(
        x=list(counts_complete.keys()),
        y=list(counts_complete.values()),
        labels={'x': 'Level', 'y': 'Cantidad de registros'},
        title='Cantidad de registros por Level',
        color=list(counts_complete.keys()),
        color_discrete_map=color_map,
    )

    fig.update_layout(
        yaxis=dict(
            dtick="auto",
            tickformat="~s",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            automargin=True
        ),
        margin=dict(t=40, b=40, l=60, r=20)
    )

    table = dash_table.DataTable(
        data=dff.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in dff.columns],
        page_size=15,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'minWidth': '100px', 'whiteSpace': 'normal'},
        filter_action='native',
        sort_action='native',
    )

    layout = [
        html.Div(f"Modelo ejecutado: {'TF-IDF + Regresión Logística' if model_type == 'tfidf' else 'Word2Vec + Regresión Logística'}", 
                 className="mb-2 fw-bold"),
        dcc.Graph(figure=fig, style={'height': '300px'}),
        html.Hr(),
        dcc.Download(id="download-data"),
        html.Button("Descargar CSV", id="btn-download", className="btn btn-primary mb-3"),
        table
    ]

    return message, layout, options, filter_value, contents
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

if __name__ == '__main__':
    app.run(debug=True)
