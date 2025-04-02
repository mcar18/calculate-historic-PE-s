import yfinance as yf
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Get daily stock price data (adjusted for splits)
def get_stock_data(ticker, start_date='2010-01-01', end_date=None):
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df = df.reset_index()  # Make 'Date' a column
    return df

# 2. Fetch quarterly income statements from financialdatasets.ai
def fetch_quarterly_income_statements(ticker, api_key):
    # Request quarterly data using period=quarter
    url = f"https://api.financialdatasets.ai/financials/income-statements?ticker={ticker}&period=quarter"
    headers = {"X-API-KEY": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        key = "income_statements"  # API returns data with underscores
        if key in data and data[key]:
            df = pd.DataFrame(data[key])
            # Convert 'report_period' to datetime
            df['report_period'] = pd.to_datetime(df['report_period'])
            df.sort_values('report_period', inplace=True)
            df = df.set_index('report_period')
            # Keep only the columns we need: net income and weighted average shares
            df = df[['net_income', 'weighted_average_shares']]
            df.rename(columns={'net_income': 'NetIncome', 
                               'weighted_average_shares': 'Shares'}, inplace=True)
            # Debug: print the first and last 5 rows of the quarterly data
            print(f"\nQuarterly income data for {ticker} (first 5 rows):")
            print(df.head(5))
            print(f"\nQuarterly income data for {ticker} (last 5 rows):")
            print(df.tail(5))
            return df
        else:
            print(f"No quarterly income statements found for {ticker}.")
            return None
    else:
        print(f"Error {response.status_code}: Unable to fetch quarterly income statements for {ticker}.")
        return None

# 3. Calculate TTM EPS from the quarterly data using a 4-quarter rolling window
def calculate_ttm_eps(quarterly_df):
    if quarterly_df is None or quarterly_df.empty:
        return None
    quarterly_df['TTM_NetIncome'] = quarterly_df['NetIncome'].rolling(window=4).sum()
    quarterly_df['TTM_Shares'] = quarterly_df['Shares'].rolling(window=4).mean()
    quarterly_df['TTM_EPS'] = quarterly_df['TTM_NetIncome'] / quarterly_df['TTM_Shares']
    # Drop rows where TTM_EPS is not available (first 3 quarters)
    ttm_df = quarterly_df.dropna(subset=['TTM_EPS'])
    return ttm_df[['TTM_EPS']]

# 4. Align the TTM EPS series with daily stock prices via merge_asof
def align_eps_with_prices(daily_df, ttm_eps_df):
    if ttm_eps_df is None or ttm_eps_df.empty:
        return None
    # Reset indexes to merge on date columns
    ttm_eps_df = ttm_eps_df.reset_index()  # 'report_period' column
    daily_df = daily_df.copy()
    # Convert both date columns to timezone-naive datetime
    daily_df['Date'] = pd.to_datetime(daily_df['Date']).dt.tz_localize(None)
    ttm_eps_df['report_period'] = pd.to_datetime(ttm_eps_df['report_period']).dt.tz_localize(None)
    # Merge the daily prices with the TTM EPS data using merge_asof (backward direction)
    merged = pd.merge_asof(daily_df.sort_values('Date'),
                           ttm_eps_df.sort_values('report_period'),
                           left_on='Date', right_on='report_period',
                           direction='backward')
    # Forward-fill any missing TTM_EPS values
    merged['TTM_EPS'] = merged['TTM_EPS'].ffill()
    merged['PE_Ratio'] = merged['Close'] / merged['TTM_EPS']
    merged.set_index('Date', inplace=True)
    return merged

# 5. Plot the daily P/E ratios for multiple tickers
def plot_pe_ratios(ticker_pe_dict):
    plt.figure(figsize=(12, 8))
    for ticker, df in ticker_pe_dict.items():
        if df is not None and 'PE_Ratio' in df.columns:
            plt.plot(df.index, df['PE_Ratio'], label=ticker)
    plt.xlabel('Date')
    plt.ylabel('P/E Ratio')
    plt.title('Historical Trailing Twelve Month P/E Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()

# 6. Main function to process multiple tickers
def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Add additional tickers here if needed
    api_key = '06f6f5be-09ae-4dc9-b133-d2f61162258d'  # Replace with your actual API key
    ticker_pe_dict = {}
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        # Retrieve daily stock price data
        daily_prices = get_stock_data(ticker, start_date='2010-01-01')
        # Fetch quarterly income statements
        quarterly_income = fetch_quarterly_income_statements(ticker, api_key)
        if quarterly_income is None:
            print(f"Skipping {ticker} due to missing financial data.")
            continue
        # Calculate TTM EPS from quarterly data
        ttm_eps_df = calculate_ttm_eps(quarterly_income)
        if ttm_eps_df is None or ttm_eps_df.empty:
            print(f"Skipping {ticker} due to insufficient quarterly data for TTM EPS.")
            continue
        # Align the TTM EPS with daily price data
        merged_df = align_eps_with_prices(daily_prices, ttm_eps_df)
        if merged_df is None:
            print(f"Skipping {ticker} due to error in aligning EPS data.")
            continue
        ticker_pe_dict[ticker] = merged_df
        print(f"Latest P/E for {ticker}: {merged_df['PE_Ratio'].iloc[-1]}")
    
    # Plot the P/E ratios for all tickers
    plot_pe_ratios(ticker_pe_dict)

if __name__ == "__main__":
    main()