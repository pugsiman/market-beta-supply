#!/usr/bin/env python3

import yfinance as yf
import numpy as np
import pandas as pd
import os
from beta import Beta

BENCHMARK_INDEX = 'SPY'
INITIAL_DATE = '1/1/2021'
NASDAQ_FTP_PATH = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt'


def create_beta_distribution(sample_returns, date_str: str, tickers: str) -> str:
    """Creates a data distribution for all current beta estimator values and residuals for the date, then saves it
    Parameters
    ----------
    sample_returns: dataframe
    date_str: string formatted as '%Y-%m-%d'
    tickers: string containing comma-seperated tickers

    Returns
    -------
    filepath: string
    """
    filepath = f'data/beta_dist-{date_str}.json'

    if not os.path.exists(filepath):
        with open(filepath, 'w+') as f:
            trailing_window = sample_returns.loc[
                date_str - pd.DateOffset(days=252) : date_str
            ]

            welch_betas = {'values': {}, 'residuals': {}}
            for ticker in tickers.split(' '):
                try:
                    beta = Beta(
                        trailing_window[BENCHMARK_INDEX], trailing_window[ticker]
                    ).welch()

                    welch_betas['values'][ticker] = beta[1]
                    welch_betas['residuals'][ticker] = beta[2]
                except (KeyError, np.linalg.LinAlgError):
                    print(
                        f'{ticker} ({date_str}) was truncated out of dataframe and could not be calculated'
                    )
                    continue

            welch_series = pd.Series(welch_betas)
            f.write(welch_series.to_json())

    return filepath


def persist_tickers(tickers) -> str:
    filepath = 'data/tickers.txt'
    if not os.path.exists(filepath) or (
        pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(filepath), unit='s')
    ) > pd.Timedelta(weeks=1):
        with open(filepath, 'w') as f:
            f.write(' '.join(map(str, tickers)))

    return filepath


def main():
    stocks_filepath = NASDAQ_FTP_PATH
    df = pd.read_csv(stocks_filepath, sep='|')
    # attempt to clean off some of the obviously bad tickers (ETFs, ETNs, tests, broken symbols etc')
    sample_stocks = df[
        (df['Test Issue'] == 'N')
        & (df['ETF'] == 'N')
        & (df['Symbol'].str.match(r'^[A-Za-z]{1,4}$', na=False))
        & ~(df['Security Name'].str.contains('ETN', na=False))
        & ~(df['Security Name'].str.contains('Acquisition', na=False))
        & ~(df['Security Name'].str.contains('ADR', na=False))
        & ~(df['Security Name'].str.contains('Depositary', na=False))
        & ~(df['Security Name'].str.contains('Trust', na=False))
    ]
    tickers = sample_stocks['Symbol'].values
    tickers_filepath = persist_tickers(tickers)
    tickers = open(tickers_filepath, 'r').read()

    ticker_list = tickers.split() + [BENCHMARK_INDEX]
    cache_path = 'data/prices_cache.parquet'
    start_date = '2020-01-01'
    batch_size = 500

    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        last_date = cached.index.max()
        new_tickers = [t for t in ticker_list if t not in cached.columns]
        next_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        frames = [cached]
        today = pd.Timestamp.now().normalize()

        if new_tickers:
            print(f'Fetching {len(new_tickers)} new tickers from {start_date}')
            for i in range(0, len(new_tickers), batch_size):
                batch = new_tickers[i : i + batch_size]
                df = yf.download(batch, start=start_date, interval='1d', auto_adjust=True, threads=False)
                if not df.empty:
                    frames.append(df['Close'] if len(batch) > 1 else df['Close'].to_frame(batch[0]))

        if pd.Timestamp(next_date) <= today:
            print(f'Fetching new dates from {next_date}')
            for i in range(0, len(ticker_list), batch_size):
                batch = ticker_list[i : i + batch_size]
                df = yf.download(batch, start=next_date, interval='1d', auto_adjust=True, threads=False)
                if not df.empty:
                    frames.append(df['Close'] if len(batch) > 1 else df['Close'].to_frame(batch[0]))
        else:
            print('Cache is already up to date')

        sample_data = pd.concat(frames)
        sample_data = sample_data.groupby(sample_data.index).first()
    else:
        print('No cache found, downloading full history')
        frames = []
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i : i + batch_size]
            df = yf.download(batch, start=start_date, interval='1d', auto_adjust=True, threads=False)
            frames.append(df['Close'])
        sample_data = pd.concat(frames, axis=1)

    sample_data.to_parquet(cache_path)

    sample_returns = sample_data.pct_change(fill_method=None).dropna(axis='index', how='all')
    sample_returns.index = pd.DatetimeIndex(sample_returns.index).tz_localize(None)
    dates = pd.bdate_range(
        start=INITIAL_DATE, end=pd.to_datetime('now').tz_localize('EST').date()
    )

    for i, date_str in enumerate(dates):
        print(f'[{i+1}/{len(dates)}] {date_str.date()}', end='\r')
        create_beta_distribution(sample_returns, date_str, tickers)
    print()


if __name__ == '__main__':
    main()
