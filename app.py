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

#df_template = pd.read_csv('dataframe_template.csv',index_col=0)

#icu_pipe = joblib.load("assets/elasticnet_feature_selection5_oversample_ICU admission_sigmoid_calibration.pickle")
#mortality_pipe = joblib.load("assets/elasticnet_feature_selection5_oversample_In-Hospital Mortality_isotonic_calibration.pickle")
#lvef_pipe = joblib.load("assets/elasticnet_feature_selection5_oversample_lvef_abnormal_sigmoid_calibration.pickle")

app = Dash(external_stylesheets=[dbc.themes.LUMEN,dbc.icons.FONT_AWESOME])

server = app.server

MIN_AGE = 18
MAX_AGE = 105
MIN_HR = 30
MAX_HR = 180
MIN_SBP = 30
MAX_SBP = 180


controls = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Label("Age",class_name="bold", width=3),
                dbc.Col(
                    [dbc.Input(
                        id='age-input', 
                        type='number',
                        placeholder="Between {}-{}".format(MIN_AGE,MAX_AGE),
                        #max=MAX_AGE,
                        #min=MIN_AGE
                ),
                #dbc.FormFeedback(
                #    "Age must be between {} and {}.".format(MIN_AGE,MAX_AGE),
                #    type="invalid",
                #),
                ]),
            ],
            className='mb-3'
        ),
        dbc.Row(
            [
                dbc.Label("Starting HR", width=3),
                dbc.Col(
                    dbc.Input(
                        id='hr-input', 
                        type='number',
                        placeholder="Between {}-{}".format(MIN_HR,MAX_HR),
                        #max=180,
                        #min=30
                )),
            ],
            className='mb-3'
        ),
        dbc.Row(
            [
                dbc.Label("Starting SBP", width=3),
                dbc.Col(
                    dbc.Input(
                        id='sbp-input', 
                        type='number',
                        placeholder="Between {}-{}".format(MIN_SBP,MAX_SBP),
                        #max=180,
                        #min=30
                )),
            ],
            className='mb-3'
        ),
        dbc.Row(
            [
                dbc.Label("Culprit Vessel", width=5),
                dbc.RadioItems(
                    id='ccv-input',
                    options=['RCA', 'LCx', 'LAD'], 
                    value='RCA',
                    inline=True
                ),
            ],
            className='mb-3'
        ),
        
        html.Div(
            [
                dbc.Label('Smoking History'),
                dbc.RadioItems(
                    id='smoking-input',
                    options=['Never', 'Ex-', 'Current'], 
                    value='Never',
                    inline=True
                ),
            ],
            className='mb-3'
        ),
        html.Div(
            [
                dbc.Label('TIMI pre-flow'),
                dbc.RadioItems(
                    id='timi-preflow-input',
                    options = ['0', '1', '2','3'],
                    value= '0',
                    inline=True
                ),
            ],
            className='mb-3'
        ),
        html.Div(
            [
                dbc.Label('Presence of Rentrop'),
                dbc.Switch(
                    id='rentrop-input',
                    value=False,
                ),
            ],
            className='mb-3'
        ),
        html.Div(
            [
                dbc.Label('Pre-hospital Arrest'),
                dbc.Switch(
                    id='prehospital-input',
                    value=False,
                ),
            ],
            className='mb-3'
        ),
        html.Div(
            [
                dbc.Label('Family history of xxx'),
                dbc.Switch(
                    id="familyhistory-input",
                    value=False,
                )
            ],
            className='mb-3'
        ),
        html.Div(
            [
                dbc.Label('Hypercholesterolaemia'),
                dbc.Switch(
                    id='hypercholesterolaemia-input',
                    value=False,
                )
            ],
            className='mb-3'
        ),
        html.Hr(),
        html.Div(
            [
               dbc.Button(
                "Calculate", id="example-button", className="d-grid gap-2 col-6 mx-auto", color='primary',n_clicks=0
        ),
            ]
        ),
    ],
    body=True,
)

icu_result = dbc.Card(
    dbc.CardBody(
        [
            html.H6("ICU Admission", className="card-subtitle mb-2"),
            dbc.Progress(
                value=5, id="icu-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            )
        ]
    ),
    className="mb-3"
)

mortality_result = dbc.Card(
    dbc.CardBody(
        [
            html.H6("In-Hospital Mortality", className="card-subtitle mb-2"),
            dbc.Progress(
                value=5, id="mortality-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            )
        ]
    ),
    className="mb-3"
)


lvef_result = dbc.Card(
    dbc.CardBody(
        [
            html.H6("LVEF < 40%", className="card-subtitle mb-2"),
            dbc.Progress(
                value=5, id="lvef-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            )
        ]
    ),
    className="mb-3"
)

risk_score = dbc.Card(
    [
        dbc.CardBody([
            html.H4("Predicted Probability", className="card-title"),
            html.Hr(),
            mortality_result,
            icu_result,
            lvef_result
        ])
    ]
)

offcanvas = html.Div(
    [
        dbc.Button(id="open-offcanvas", n_clicks=0,children=html.I(className = "fa-solid fa-circle-info fa-xl"),
                style={'color':'white'}),
        dbc.Offcanvas(
            [
                html.P(
                    "Some description of the project. "
                ),
                html.P(
                    [
                        "Paper  ", html.A("link", href="https://github.com/harrisonnguyen/",)
                    ]
                ),
                html.P(
                    [
                        "Code of the metholody and application can be found on ", 
                        html.A("github", href="https://github.com/harrisonnguyen/stemi_risk_prediction")
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
        offcanvas
    ],
    brand="Risk Prediction",
    color="primary",
    dark=True,
    class_name='mb-3',
    style={'border-radius': '10px'}
)

app.layout = html.Div(
    [

    dbc.Container(
    [
        navbar,
        dbc.Row(
            [
                dbc.Col(md=1),
                dbc.Col(controls, md=4),
                dbc.Col([
                    risk_score
                ], md=4),
                #dbc.Col(dcc.Graph(id='graph-content'), md=4)

            ],
            align="center",
        ),
        #html.Hr(),
        

    ],
    fluid='md',
    ),
    dbc.Container([
    dbc.Row(
            [
                dbc.Col(dbc.Table(id='table-content', striped=True, bordered=True, hover=True), md=12),
            ],
            align="center",
        )
    ],
    fluid=True)
    ]

)


@callback(
    #Output('table-content', 'children'),
    Output('icu-prob', 'value'),
    Output('icu-prob', 'label'),
    Output('mortality-prob', 'value'),
    Output('mortality-prob', 'label'),
    Output('lvef-prob', 'value'),
    Output('lvef-prob', 'label'),
    Input('example-button', 'n_clicks'),
    State('age-input', 'value'),
    State('ccv-input', 'value'),
    State('hr-input', 'value'),
    State('sbp-input', 'value'),
    State('smoking-input', 'value'),
    State('timi-preflow-input', 'value'),
    State('rentrop-input', 'value'),
    State('prehospital-input', 'value'),
    State('familyhistory-input', 'value'),
    State('hypercholesterolaemia-input', 'value'),
    prevent_initial_call=True
)
def predict_risk(n_clicks,age,ccv,hr,sbp,smoking,timi,rentrop,prehospital,family_hist,hyperchol):
    first_row = 0
    df_template.loc[first_row,'Age'] = age
    df_template.loc[first_row,'Starting HR'] = hr
    df_template.loc[first_row,'Starting SBP'] = sbp

    if check_age_validity(age) or  check_hr_validity(hr) or  check_sbp_validity(sbp):
        return (
            5,
            "",
            5,
            "",
            5,
            ""
    )

    # dummy variables
    if ccv == 'LCx':
        df_template.loc[first_row,'Coded Cuplrit Vessel (RCA = 1, LCx = 2, LAD = 3)_2.0'] = 1
        df_template.loc[first_row, 'Coded Cuplrit Vessel (RCA = 1, LCx = 2, LAD = 3)_3.0'] = 0
    elif ccv == 'LAD':
        df_template.loc[first_row, 'Coded Cuplrit Vessel (RCA = 1, LCx = 2, LAD = 3)_2.0'] = 0
        df_template.loc[first_row, 'Coded Cuplrit Vessel (RCA = 1, LCx = 2, LAD = 3)_3.0'] = 1
    else:
        df_template.loc[first_row, 'Coded Cuplrit Vessel (RCA = 1, LCx = 2, LAD = 3)_2.0'] = 0
        df_template.loc[first_row, 'Coded Cuplrit Vessel (RCA = 1, LCx = 2, LAD = 3)_3.0'] = 0


    if smoking == 'Ex-':
        df_template.loc[first_row, 'Smoking History( 2= current, 1 = ex smoke, 0 = never)_1.0'] = 1
        df_template.loc[first_row, 'Smoking History( 2= current, 1 = ex smoke, 0 = never)_2.0'] = 0
    elif smoking == 'Current':
        df_template.loc[first_row, 'Smoking History( 2= current, 1 = ex smoke, 0 = never)_1.0'] = 0
        df_template.loc[first_row, 'Smoking History( 2= current, 1 = ex smoke, 0 = never)_2.0'] = 1
    else:
        df_template.loc[first_row, 'Smoking History( 2= current, 1 = ex smoke, 0 = never)_1.0'] = 0
        df_template.loc[first_row, 'Smoking History( 2= current, 1 = ex smoke, 0 = never)_2.0'] = 0

    if timi == '1':
         df_template.loc[first_row, 'TIMI flow pre_1.0'] = 1
         df_template.loc[first_row, 'TIMI flow pre_2.0'] = 0
         df_template.loc[first_row, 'TIMI flow pre_3.0'] = 0
    elif timi == '2':
        df_template.loc[first_row, 'TIMI flow pre_1.0'] = 0
        df_template.loc[first_row, 'TIMI flow pre_2.0'] = 1
        df_template.loc[first_row, 'TIMI flow pre_3.0'] = 0
    elif timi == '3':
        df_template.loc[first_row, 'TIMI flow pre_1.0'] = 0
        df_template.loc[first_row, 'TIMI flow pre_2.0'] = 0
        df_template.loc[first_row, 'TIMI flow pre_3.0'] = 1
    else:
        df_template.loc[first_row, 'TIMI flow pre_1.0'] = 0
        df_template.loc[first_row, 'TIMI flow pre_2.0'] = 0
        df_template.loc[first_row, 'TIMI flow pre_3.0'] = 0
    
    if rentrop:
        df_template.loc[first_row, 'Rentrop Simplified_1.0'] = 1
    else:
        df_template.loc[first_row, 'Rentrop Simplified_1.0'] = 0

    if prehospital:
        df_template.loc[first_row, 'Pre-Hospital Arrest_1.0'] = 1
    else:
        df_template.loc[first_row, 'Pre-Hospital Arrest_1.0'] = 0
    
    if family_hist:
        df_template.loc[first_row, 'Family History_1.0'] = 1
    else:
        df_template.loc[first_row, 'Family History_1.0'] = 0
    
    if hyperchol:
        df_template.loc[first_row, 'Hypercholesterolaemia_1.0'] = 1
    else:
        df_template.loc[first_row, 'Hypercholesterolaemia_1.0'] = 0

    icu_pred = round(icu_pipe.predict_proba(df_template)[0,1],2)
    icu_prob = "{:.0%}".format(icu_pred)

    mortality_pred = round(mortality_pipe.predict_proba(df_template)[0,1],2)
    mortality_prob = "{:.0%}".format(mortality_pred)

    lvef_pred = round(lvef_pipe.predict_proba(df_template)[0,1],2)
    lvef_prob = "{:.0%}".format(lvef_pred)

    return (
        #dbc.Table.from_dataframe(df_template),
        icu_pred*100,
        icu_prob,
        mortality_pred*100,
        mortality_prob,
        lvef_pred*100,
        lvef_prob
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


@app.callback(
    Output("age-input", "invalid"),
    Input("age-input", "value"),
)
def check_age_validity(value):
    if value:
        is_invalid = value < MIN_AGE or value > MAX_AGE
        return is_invalid
    return False

@app.callback(
    Output("hr-input", "invalid"),
    Input("hr-input", "value"),
)
def check_hr_validity(value):
    if value:
        is_invalid = value < MIN_HR or value > MAX_HR
        return is_invalid
    return False


@app.callback(
    Output("sbp-input", "invalid"),
    Input("sbp-input", "value"),
)
def check_sbp_validity(value):
    if value:
        is_invalid = value < MIN_SBP or value > MAX_SBP
        return is_invalid
    return False



if __name__ == '__main__':
    app.run(debug=True)