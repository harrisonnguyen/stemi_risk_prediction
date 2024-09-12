import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import joblib
import os
from dash.exceptions import PreventUpdate
dash.register_page(__name__, 
                   path='/',
                   title='STEMI-ML Score',
                   name='STEMI-ML Score')


df_template = pd.read_csv('dataframe_template.csv',index_col=0)

icu_pipe = joblib.load("model/elasticnet_feature_selection5_oversample_ICU admission_isotonic_calibration.pickle")
mortality_pipe = joblib.load("model/elasticnet_feature_selection5_oversample_In-Hospital Mortality_isotonic_calibration.pickle")
lvef_pipe = joblib.load("model/elasticnet_feature_selection5_oversample_lvef_abnormal_isotonic_calibration.pickle")

features_to_drop = ['LVEF FINAL','lvef_abnormal_1.0']

#app = Dash(external_stylesheets=[dbc.themes.LUMEN,dbc.icons.FONT_AWESOME])

#server = app.server

MIN_AGE = 18
MAX_AGE = 100
MIN_HR = 30
MAX_HR = 170
MIN_SBP = 30
MAX_SBP = 230
PROGRESS_BAR_MIN_VALUE = 4


controls = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Label("Age (years)",class_name="bold", width=5),
                dbc.Col(
                    [dbc.Input(
                        id='age-inhos-input', 
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
            className='mb-2'
        ),
        dbc.Row(
            [
                dbc.Label("Starting heart rate (bpm)", width=5),
                dbc.Col(
                    dbc.Input(
                        id='hr-input', 
                        type='number',
                        placeholder="Between {}-{}".format(MIN_HR,MAX_HR),
                        #max=180,
                        #min=30
                )),
            ],
            className='mb-2'
        ),
        dbc.Row(
            [
                dbc.Label("Starting systolic blood pressure (mmHg)", width=5),
                dbc.Col(
                    dbc.Input(
                        id='sbp-input', 
                        type='number',
                        placeholder="Between {}-{}".format(MIN_SBP,MAX_SBP),
                        #max=180,
                        #min=30
                )),
            ],
            className='mb-2'
        ),
        
        dbc.Row(
            [
                dbc.Label(
                    html.Span("Culprit vessel",id="tooltip-target",style={"textDecoration": "underline", "cursor": "help",'text-decoration-style': 'dotted','text-underline-offset':'0.3rem'}), 
                    width=5,color='primary'),
                dbc.Tooltip(html.P([
                    html.Span("RCA = right coronary artery"),
                    html.Br(),
                    html.Span("LCx = left circumflex artery"), 
                    html.Br(),
                    html.Span("LAD = left anterior descending artery")]),
                    target="tooltip-target"),
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
                dbc.Label('Smoking history'),
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
                dbc.Label(
                    'TIMI flow at onset of angiography'),
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
                dbc.Label(html.Span('Robust collateral recruitment',
                    id="rentrop-tooltip-target",
                    style={"textDecoration": "underline", "cursor": "help",'text-decoration-style': 'dotted','text-underline-offset':'0.3rem'}),
                    color='primary'),
                dbc.Tooltip(html.P([
                    html.Span("Robust collateral recruitment is Rentrop grade 2 or 3"),
                    html.Br(),
                    html.Span("Poor collateral recruitment is defined as Rentrop grade 0 or 1")]),
                    target="rentrop-tooltip-target"),
                dbc.Switch(
                    id='rentrop-input',
                    value=False,
                ),
            ],
            className='mb-2'
        ),
        html.Div(
            [
                dbc.Label('Pre-hospital arrest'),
                dbc.Switch(
                    id='prehospital-input',
                    value=False,
                ),
            ],
            className='mb-2'
        ),
        html.Div(
            [
                dbc.Label('Family history of coronary disease before 50 years'),
                dbc.Switch(
                    id="familyhistory-input",
                    value=False,
                )
            ],
            className='mb-2'
        ),
        html.Div(
            [
                dbc.Label('Hypercholesterolaemia'),
                dbc.Switch(
                    id='hypercholesterolaemia-input',
                    value=False,
                )
            ],
            className='mb-2'
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
            dbc.Label(
                html.Span("ICU Admission",
                style={"textDecoration": "underline", "cursor": "help",'text-decoration-style': 'dotted','text-underline-offset':'0.3rem'}),
                color='primary', 
                className="card-subtitle mb-2 mt-0 pt-0",id="icu-tooltip-target",
            ),
            dbc.Progress(
                value=PROGRESS_BAR_MIN_VALUE, id="icu-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            ),
            dbc.Tooltip(html.P([
                    html.Span("Intensive care unit admission during index presentation.")]),
                    target="icu-tooltip-target"),
        ]
    ),
    className="mb-3"
)

mortality_result = dbc.Card(
    dbc.CardBody(
        [
            html.H6("In-Hospital Mortality", className="card-subtitle mb-2"),
            dbc.Progress(
                value=PROGRESS_BAR_MIN_VALUE, id="mortality-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            )
        ]
    ),
    className="mb-3"
)


lvef_result = dbc.Card(
    dbc.CardBody(
        [
            dbc.Label(
                html.Span("LVEF < 40%",
                style={"textDecoration": "underline", "cursor": "help",'text-decoration-style': 'dotted','text-underline-offset':'0.3rem'}),
                color='primary', 
                className="card-subtitle mb-2 mt-0 pt-0",id="lvef-pred-tooltip-target",
            ),
            dbc.Progress(
                value=PROGRESS_BAR_MIN_VALUE, id="lvef-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
            ),
              dbc.Tooltip(html.P([
                    html.Span("Left ventricular ejection fraction less than 40% on index presentation.")]),
                    target="lvef-pred-tooltip-target"),
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



layout = html.Div(
    [

    dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(lg=1),
                dbc.Col(controls, lg=5),
                dbc.Col([
                    risk_score
                ], lg=5),
                #dbc.Col(dcc.Graph(id='graph-content'), md=4)

            ],
            align="center",
        ),
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


@callback(
    #Output('table-content', 'children'),
    Output('icu-prob', 'value'),
    Output('icu-prob', 'label'),
    Output('mortality-prob', 'value'),
    Output('mortality-prob', 'label'),
    Output('lvef-prob', 'value'),
    Output('lvef-prob', 'label'),
    Input('example-button', 'n_clicks'),
    State('age-inhos-input', 'value'),
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

    if check_age_validity(age,n_clicks) or  check_hr_validity(hr) or  check_sbp_validity(sbp):
        return (
            PROGRESS_BAR_MIN_VALUE,
            "",
            PROGRESS_BAR_MIN_VALUE,
            "",
            PROGRESS_BAR_MIN_VALUE,
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

    icu_pred = round(icu_pipe.predict_proba(df_template.drop(features_to_drop,axis=1))[0,1],2)
    icu_prob = "{:.0%}".format(icu_pred)

    mortality_pred = round(mortality_pipe.predict_proba(df_template.drop(features_to_drop,axis=1))[0,1],2)
    mortality_prob = "{:.0%}".format(mortality_pred)

    lvef_pred = round(lvef_pipe.predict_proba(df_template.drop(features_to_drop,axis=1))[0,1],2)
    lvef_prob = "{:.0%}".format(lvef_pred)


    if icu_pred < PROGRESS_BAR_MIN_VALUE/100:
        icu_pred = PROGRESS_BAR_MIN_VALUE/100
    if mortality_pred < PROGRESS_BAR_MIN_VALUE/100:
        mortality_pred = PROGRESS_BAR_MIN_VALUE/100
    if lvef_pred < PROGRESS_BAR_MIN_VALUE/100:
        lvef_pred = PROGRESS_BAR_MIN_VALUE/100
    print(icu_pred)
    print(mortality_pred)
    print(lvef_pred)
    print(df_template[['Age','LVEF FINAL','Coded Treatment (1 = PCI, 2 = PTCA, 3 = emergent CABG, 4 = med)_2.0']])


    return (
        #dbc.Table.from_dataframe(df_template),
        icu_pred*100,
        icu_prob,
        mortality_pred*100,
        mortality_prob,
        lvef_pred*100,
        lvef_prob
    )


@callback(
    Output("age-inhos-input", "invalid"),
    Input("age-inhos-input", "value"),
    Input('example-button', 'n_clicks'),
)
def check_age_validity(value,n_clicks):
    if n_clicks == 0:
        raise PreventUpdate 
    if value:
        is_invalid = value < MIN_AGE or value > MAX_AGE
        return is_invalid
    elif value is None:
        return True
    return False

@callback(
    Output("hr-input", "invalid"),
    Input("hr-input", "value")
)
def check_hr_validity(value):
    if value:
        is_invalid = value < MIN_HR or value > MAX_HR
        return is_invalid
    return False


@callback(
    Output("sbp-input", "invalid"),
    Input("sbp-input", "value")
)
def check_sbp_validity(value):
    if value:
        is_invalid = value < MIN_SBP or value > MAX_SBP
        return is_invalid
    return False



#if __name__ == '__main__':
#    app.run(debug=True)
