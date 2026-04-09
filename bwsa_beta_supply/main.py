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

    figs.data[0].name = 'Supply Rate'
    figs.data[1].name = 'Dispersion (Q90 − Q10)'

    # Shade supply zones
    supply = beta_supply['supply_count']
    dates = pd.to_datetime(supply.index)

    def add_shaded_regions(fig, dates, values, threshold, direction, color):
        in_region = False
        start = None
        for i, (dt, val) in enumerate(zip(dates, values)):
            active = val < threshold if direction == 'below' else val >= threshold
            if active and not in_region:
                start = dt
                in_region = True
            elif not active and in_region:
                fig.add_vrect(x0=start, x1=dates[i - 1], fillcolor=color,
                              opacity=0.15, line_width=0, layer='below')
                in_region = False
        if in_region:
            fig.add_vrect(x0=start, x1=dates[-1], fillcolor=color,
                          opacity=0.15, line_width=0, layer='below')

    add_shaded_regions(figs, dates, supply.values, 2.0, 'below', 'green')
    add_shaded_regions(figs, dates, supply.values, 3.75, 'above', 'red')

    # Squeeze markers: supply < 2% AND dispersion at/below 20d trend
    disp_series = beta_dispersion.sort_index()
    disp_series.index = pd.to_datetime(disp_series.index)
    disp_trend = disp_series.rolling(20).mean()
    disp_dev = disp_series - disp_trend

    squeeze_dates = []
    prev_squeeze = False
    for dt, sr in zip(dates, supply.values):
        is_squeeze = sr < 2.0 and dt in disp_dev.index and pd.notna(disp_dev.loc[dt]) and disp_dev.loc[dt] <= 0
        if is_squeeze and not prev_squeeze:
            squeeze_dates.append(dt)
        prev_squeeze = is_squeeze

    if squeeze_dates:
        squeeze_supply = supply.loc[[d.strftime('%Y-%m-%d') for d in squeeze_dates if d.strftime('%Y-%m-%d') in supply.index]]
        figs.add_trace(go.Scatter(
            x=squeeze_dates,
            y=[supply.loc[d.strftime('%Y-%m-%d')] if d.strftime('%Y-%m-%d') in supply.index else None for d in squeeze_dates],
            mode='markers',
            marker=dict(symbol='triangle-up', size=8, color='yellow', opacity=0.7),
            name='Squeeze Setup',
            hovertemplate='Date: %{x}<br>Supply: %{y:.2f}%<br>',
        ))

    figs.update_traces(
        hovertemplate='Date: %{x}<br>Value: %{y:.2f}<br>',
        selector=dict(mode='lines'),
    )

    figs.show()


if __name__ == '__main__':
    main()
