import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import joblib
import os
from dash.exceptions import PreventUpdate

dash.register_page(__name__, 
                   path='/mortality',
                   title='STEMI-ML Score 1-Year Mortality',
                   name='STEMI-ML Score 1-Year Mortality')

df_template = pd.read_csv('dataframe_template.csv',index_col=0)
mortality_year_pipe = joblib.load("model/elasticnet_feature_selection5_Outcome_sigmoid_calibration.pickle")


features_to_drop = ['LVEF FINAL','lvef_abnormal_1.0']

#app = Dash(external_stylesheets=[dbc.themes.LUMEN,dbc.icons.FONT_AWESOME])

#server = app.server

MIN_AGE = 18
MAX_AGE = 100
MIN_LVEF = 10
MAX_LVEF = 60
PROGRESS_BAR_MIN_VALUE = 4


controls = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Label("Age (years)",class_name="bold", width=5),
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
        dbc.Row(
            [
                dbc.Label("LVEF (%)", width=5),
                dbc.Col(
                    dbc.Input(
                        id='lvef-input', 
                        type='number',
                        placeholder="Between {}-{}".format(MIN_LVEF,MAX_LVEF),
                )),
            ],
            className='mb-2'
        ),
        html.Div(
            [
                dbc.Label('Balloon angioplasty'),
                dbc.Switch(
                    id='ptca-input',
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

mortality_year_result = dbc.Card(
    dbc.CardBody(
        [
            html.H6("1 Year Mortality", className="card-subtitle mb-2"),
            dbc.Progress(
                value=PROGRESS_BAR_MIN_VALUE, id="mortality-year-prob", animated=True, striped=True,style={"height": "20px"}, color='primary'
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
            mortality_year_result
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
    Output('mortality-year-prob', 'value'),
    Output('mortality-year-prob', 'label'),
    Input('example-button', 'n_clicks'),
    State('age-input', 'value'),
    State('lvef-input', 'value'),
    State('prehospital-input', 'value'),
    State('familyhistory-input', 'value'),
    State('ptca-input', 'value'),
    prevent_initial_call=True
)
def predict_risk(n_clicks,age,lvef,prehospital,family_hist,ptca):
    first_row = 0
    df_template.loc[first_row,'Age'] = age
    df_template.loc[first_row,'LVEF FINAL'] = lvef

    if check_age_validity(age,n_clicks) or  check_lvef_validity(lvef,n_clicks):
        return (
            PROGRESS_BAR_MIN_VALUE,
            ""
    )



    if prehospital:
        df_template.loc[first_row, 'Pre-Hospital Arrest_1.0'] = 1
    else:
        df_template.loc[first_row, 'Pre-Hospital Arrest_1.0'] = 0
    
    if family_hist:
        df_template.loc[first_row, 'Family History_1.0'] = 1
    else:
        df_template.loc[first_row, 'Family History_1.0'] = 0
    
    if ptca:
        df_template.loc[first_row, 'Coded Treatment (1 = PCI, 2 = PTCA, 3 = emergent CABG, 4 = med)_2.0'] = 1
        df_template.loc[first_row, 'Coded Treatment (1 = PCI, 2 = PTCA, 3 = emergent CABG, 4 = med)_3.0'] = 0
        df_template.loc[first_row, 'Coded Treatment (1 = PCI, 2 = PTCA, 3 = emergent CABG, 4 = med)_4.0'] = 0
    else:
        df_template.loc[first_row, 'Coded Treatment (1 = PCI, 2 = PTCA, 3 = emergent CABG, 4 = med)_2.0'] = 0
        df_template.loc[first_row, 'Coded Treatment (1 = PCI, 2 = PTCA, 3 = emergent CABG, 4 = med)_3.0'] = 0
        df_template.loc[first_row, 'Coded Treatment (1 = PCI, 2 = PTCA, 3 = emergent CABG, 4 = med)_4.0'] = 0

    
    mortality_year_pred = round(mortality_year_pipe.predict_proba(df_template)[0,1],2)
    mortality_year_prob = "{:.0%}".format(mortality_year_pred)

   
    if mortality_year_pred < PROGRESS_BAR_MIN_VALUE/100:
        mortality_year_pred = PROGRESS_BAR_MIN_VALUE/100
    print(mortality_year_pred)
    print(df_template[['Age','LVEF FINAL','Coded Treatment (1 = PCI, 2 = PTCA, 3 = emergent CABG, 4 = med)_2.0']])


    return (
        #dbc.Table.from_dataframe(df_template),
        mortality_year_pred*100,
        mortality_year_prob
    )



@callback(
    Output("age-input", "invalid"),
    Input("age-input", "value"),
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
    Output("lvef-input", "invalid"),
    Input("lvef-input", "value"),
    Input('example-button', 'n_clicks'),
)
def check_lvef_validity(value,n_clicks):
    if n_clicks == 0:
        raise PreventUpdate 
    if value:
        is_invalid = value < MIN_LVEF or value > MAX_LVEF
        return is_invalid
    elif value is None:
        return True
    return False



#if __name__ == '__main__':
#    app.run(debug=True)
