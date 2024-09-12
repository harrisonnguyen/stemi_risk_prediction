import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import joblib
import os



#dir = 'C:\\Users\\Harrison Nguyen\\Documents\\CardiacCovidModel\\stemi\\'
#df_template = pd.read_csv(os.path.join(dir,'app\\dataframe_template.csv'),index_col=0)
#icu_pipe = joblib.load(os.path.join(dir,"model\\elasticnet_feature_selection5_oversample_ICU admission_sigmoid_calibration.pickle"))
#mortality_pipe = joblib.load(os.path.join(dir,"model\\elasticnet_feature_selection5_oversample_In-Hospital Mortality_isotonic_calibration.pickle"))
#lvef_pipe = joblib.load(os.path.join(dir,"model\\elasticnet_feature_selection5_oversample_lvef_abnormal_sigmoid_calibration.pickle"))

df_template = pd.read_csv('dataframe_template.csv',index_col=0)

icu_pipe = joblib.load("model/elasticnet_feature_selection5_oversample_ICU admission_isotonic_calibration.pickle")
mortality_pipe = joblib.load("model/elasticnet_feature_selection5_oversample_In-Hospital Mortality_isotonic_calibration.pickle")
lvef_pipe = joblib.load("model/elasticnet_feature_selection5_oversample_lvef_abnormal_isotonic_calibration.pickle")
mortality_year_pipe = joblib.load("model/elasticnet_feature_selection5_Outcome_sigmoid_calibration.pickle")

features_to_drop = ['LVEF FINAL','lvef_abnormal_1.0']

app = Dash(use_pages=True,external_stylesheets=[dbc.themes.LUMEN,dbc.icons.FONT_AWESOME])

server = app.server





offcanvas = html.Div(
    [
        dbc.Button(id="open-offcanvas", n_clicks=0,children=html.I(className = "fa-solid fa-circle-info fa-xl"),
                style={'color':'white'}),
        dbc.Offcanvas(
            [
                html.P(
                    """
                        The STEMI-ML score is a machine-learning based risk prediction score for in-hospital mortality, intensive care unit admission, left ventricular ejection fraction less than 40% & 1-year mortality in STEMI patients. This score has been derived from a cohort of 1863 consecutive, STEMI patients at single, tertiary Australian centre who underwent primary percutaneous coronary intervention or rescue percutaneous coronary intervention from 2010 to 2019.
                    """
                ),
                #html.P(
                #    [
                #        "This work is based on a paper found in this ", html.A("link", href="",)
                #    ]
                #),
                html.P(
                    [
                        "Code of the methodology and application can be found on ", 
                        html.A("github", href="https://github.com/harisritharan/stemi_risk_prediction")
                    ]
                )
                ],
            id="offcanvas",
            title="Information",
            is_open=False,
        ),
    ]
)


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("In-hospital outcomes", href="/")),
        dbc.NavItem(dbc.NavLink("1-year mortality", href="/mortality")),
        offcanvas
    ],
    brand="STEMI-ML Score",
    brand_href="/",
    color="primary",
    dark=True,
    class_name='mb-3',
    style={'border-radius': '10px'},
    links_left=False,
    fluid=True
)

app.layout = html.Div(
    [

    dbc.Container(
    [
        navbar,
        dash.page_container
        #html.Hr(),
        

    ],
    fluid='sm',
    ),
    dbc.Container([
    dbc.Row(
            [
                dbc.Col(dbc.Table(id='table-content', striped=True, bordered=True, hover=True), lg=12),
            ],
            align="center",
        )
    ],
    fluid=True)
    ]

)

@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open



if __name__ == '__main__':
    app.run(debug=True)
