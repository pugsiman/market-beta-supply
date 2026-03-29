#!/usr/bin/env python3

import json

import pandas as pd
import plotly.graph_objects as go


import os
import re

pd.options.plotting.backend = 'plotly'


def main():
    directory = os.fsencode('data')
    series = []
    for file in os.scandir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.json'):
            try:
                with open(filename) as f:
                    index = re.search('(?<=-).+(?= )', f.name).group()  # extract date
                    data = json.load(f)
                    # backwards compatiability for data sets that didn't have residuals calculated
                    if 'values' in data:
                        json_series = pd.Series(data['values'])
                    else:
                        json_series = pd.Series(data)

                    series.append(json_series.rename(index))
            except ValueError:
                continue

    df = pd.DataFrame(series).sort_index().T
    beta_supply = ((df[df > 1.9].count() / df.count()) * 100).to_frame(
        name='supply_count'
    )
    beta_dispersion = (
        df.where(df.gt(df.quantile(0.9))).stack().groupby(level=1).agg('mean')
    ) - df.where(df.lt(df.quantile(0.1))).stack().groupby(level=1).agg('mean')
    beta_dispersion_trace = beta_dispersion.to_frame(name='beta_dispersion').plot()

    figs = (
        go.Figure(
            data=beta_supply['supply_count'].plot().data
            + beta_dispersion_trace.data
        )
        .update_layout(
            title=dict(text='Beta Supply', font_size=24, x=0.5),
            xaxis=dict(
                ticklabelmode='period',
                dtick='M1',
                showline=True,
                showgrid=False,
                type='date',
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all'),
                        ]
                    ),
                    font_color='black',
                    activecolor='gray',
                ),
                rangeslider=dict(visible=True),
            ),
            yaxis=dict(showline=True, showgrid=False),
            template='plotly_dark',
            legend=dict(x=0.1, y=1.1, orientation='h', font=dict(color='#FFFFFF')),
        )
    )

    figs.data[0].name = 'Supply Rate (β > 1.9)'
    figs.data[1].name = 'Dispersion (Q90 − Q10)'

    figs.update_traces(
        hovertemplate='Date: %{x}<br>Value: %{y:.2f}<br>',
    )

    figs.show()


if __name__ == '__main__':
    main()
