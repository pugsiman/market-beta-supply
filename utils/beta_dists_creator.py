#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from beta import Beta

BENCHMARK_INDEX = "SPY"
INITIAL_DATE = "1/1/2021"
NASDAQ_FTP_PATH = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt"
ACTIVE_TICKERS_PATH = Path("data/tickers.txt")
RETAINED_TICKERS_PATH = Path("data/retained_tickers.txt")
TICKER_REFRESH_AGE = pd.Timedelta(weeks=1)


def merge_tickers(*ticker_groups) -> list[str]:
    merged = []
    seen = set()

    for ticker_group in ticker_groups:
        for ticker in ticker_group:
            ticker = str(ticker).strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            merged.append(ticker)

    return merged


def load_tickers(filepath: str | Path) -> list[str]:
    path = Path(filepath)
    if not path.exists():
        return []

    return path.read_text().split()


def persist_tickers(tickers, filepath: str | Path, refresh_after: pd.Timedelta | None = None) -> str:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    should_write = not path.exists()
    if refresh_after is not None and path.exists():
        file_age = pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")
        should_write = file_age > refresh_after

    if should_write:
        path.write_text(" ".join(map(str, tickers)))

    return str(path)


def build_retained_ticker_universe(active_tickers, persisted_tickers=(), cached_tickers=()) -> list[str]:
    return merge_tickers(active_tickers, persisted_tickers, cached_tickers)


def create_beta_distribution(sample_returns, date_str: str, tickers) -> str:
    """Creates a data distribution for all current beta estimator values and residuals for the date, then saves it
    Parameters
    ----------
    sample_returns: dataframe
    date_str: string formatted as '%Y-%m-%d'
    tickers: iterable containing ticker symbols

    Returns
    -------
    filepath: string
    """
    filepath = f"data/beta_dist-{date_str}.json"

    if not os.path.exists(filepath):
        with open(filepath, "w+") as f:
            trailing_window = sample_returns.loc[date_str - pd.DateOffset(days=252) : date_str]

            welch_betas = {"values": {}, "residuals": {}}
            for ticker in tickers:
                try:
                    beta = Beta(trailing_window[BENCHMARK_INDEX], trailing_window[ticker]).welch()

                    welch_betas["values"][ticker] = beta[1]
                    welch_betas["residuals"][ticker] = beta[2]
                except (KeyError, np.linalg.LinAlgError):
                    print(f"{ticker} ({date_str}) was truncated out of dataframe and could not be calculated")
                    continue

            welch_series = pd.Series(welch_betas)
            f.write(welch_series.to_json())

    return filepath


def main():
    stocks_filepath = NASDAQ_FTP_PATH
    df = pd.read_csv(stocks_filepath, sep="|")
    # attempt to clean off some of the obviously bad tickers (ETFs, ETNs, tests, broken symbols etc')
    sample_stocks = df[
        (df["Test Issue"] == "N")
        & (df["ETF"] == "N")
        & (df["Symbol"].str.match(r"^[A-Za-z]{1,4}$", na=False))
        & ~(df["Security Name"].str.contains("ETN", na=False))
        & ~(df["Security Name"].str.contains("Acquisition", na=False))
        & ~(df["Security Name"].str.contains("ADR", na=False))
        & ~(df["Security Name"].str.contains("Depositary", na=False))
        & ~(df["Security Name"].str.contains("Trust", na=False))
    ]
    active_tickers = sample_stocks["Symbol"].tolist()
    tickers_filepath = persist_tickers(active_tickers, ACTIVE_TICKERS_PATH, refresh_after=TICKER_REFRESH_AGE)
    active_tickers = load_tickers(tickers_filepath)

    ticker_list = merge_tickers(active_tickers, [BENCHMARK_INDEX])
    cache_path = "data/prices_cache.parquet"
    start_date = "2020-01-01"
    batch_size = 500

    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        last_date = cached.index.max()
        new_tickers = [t for t in ticker_list if t not in cached.columns]
        next_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        retained_tickers = build_retained_ticker_universe(
            active_tickers,
            persisted_tickers=load_tickers(RETAINED_TICKERS_PATH),
            cached_tickers=[column for column in cached.columns if column != BENCHMARK_INDEX],
        )

        frames = [cached]
        today = pd.Timestamp.now().normalize()

        if new_tickers:
            print(f"Fetching {len(new_tickers)} new tickers from {start_date}")
            for i in range(0, len(new_tickers), batch_size):
                batch = new_tickers[i : i + batch_size]
                df = yf.download(batch, start=start_date, interval="1d", auto_adjust=True, threads=False)
                if not df.empty:
                    frames.append(df["Close"] if len(batch) > 1 else df["Close"].to_frame(batch[0]))

        if pd.Timestamp(next_date) < today:
            print(f"Fetching new dates from {next_date}")
            for i in range(0, len(ticker_list), batch_size):
                batch = ticker_list[i : i + batch_size]
                df = yf.download(batch, start=next_date, interval="1d", auto_adjust=True, threads=False)
                if not df.empty:
                    frames.append(df["Close"] if len(batch) > 1 else df["Close"].to_frame(batch[0]))
        else:
            print("Cache is already up to date")

        sample_data = pd.concat(frames)
        sample_data = sample_data.groupby(sample_data.index).first()
    else:
        print("No cache found, downloading full history")
        frames = []
        for i in range(0, len(ticker_list), batch_size):
            batch = ticker_list[i : i + batch_size]
            df = yf.download(batch, start=start_date, interval="1d", auto_adjust=True, threads=False)
            frames.append(df["Close"])
        sample_data = pd.concat(frames, axis=1)
        retained_tickers = build_retained_ticker_universe(active_tickers)

    sample_data.to_parquet(cache_path)
    persist_tickers(retained_tickers, RETAINED_TICKERS_PATH)

    sample_returns = sample_data.pct_change(fill_method=None).dropna(axis="index", how="all")
    sample_returns.index = pd.DatetimeIndex(sample_returns.index).tz_localize(None)
    dates = pd.bdate_range(start=INITIAL_DATE, end=pd.to_datetime("now").tz_localize("EST").date())

    for i, date_str in enumerate(dates):
        print(f"[{i+1}/{len(dates)}] {date_str.date()}", end="\r")
        create_beta_distribution(sample_returns, date_str, retained_tickers)
    print()


if __name__ == "__main__":
    main()
