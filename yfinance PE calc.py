import yfinance as yf
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

# 2. Get quarterly financials from yfinance and compute TTM EPS
def get_quarterly_ttm_eps(ticker):
    t = yf.Ticker(ticker)
    q_fin = t.quarterly_financials
    if q_fin.empty:
        print(f"Quarterly financials for {ticker} is empty.")
        return None
    # Debug: print available keys in the index
    print(f"Quarterly financials index for {ticker}: {list(q_fin.index)}")
    # Look for a row with "net income" (case-insensitive)
    net_income_key = None
    for key in q_fin.index:
        if "net income" in key.lower():
            net_income_key = key
            break
    if net_income_key is None:
        print(f"'Net Income' not found in quarterly financials for {ticker}. Available keys: {list(q_fin.index)}")
        return None
    # Extract net income and sort by quarter-end dates (columns)
    net_income = q_fin.loc[net_income_key]
    net_income = net_income.sort_index()  # Ensure ascending order by date
    # Compute TTM net income as rolling sum of 4 quarters
    ttm_net_income = net_income.rolling(window=4).sum()
    # For shares, we use the current shares outstanding (yfinance doesn't provide historical shares)
    info = t.info
    shares = info.get("sharesOutstanding", None)
    if shares is None:
        print(f"Shares outstanding not found for {ticker}.")
        return None
    # Calculate TTM EPS
    ttm_eps = ttm_net_income / shares
    # Convert to DataFrame
    df = ttm_eps.reset_index()
    df.columns = ["report_period", "TTM_EPS"]
    df.sort_values("report_period", inplace=True)
    return df

# 3. Align the TTM EPS series with daily price data using merge_asof
def align_eps_with_prices(daily_df, eps_df):
    if eps_df is None or eps_df.empty:
        return None
    eps_df = eps_df.copy()
    daily_df = daily_df.copy()
    # Ensure both date columns are timezone-naive
    daily_df['Date'] = pd.to_datetime(daily_df['Date']).dt.tz_localize(None)
    eps_df['report_period'] = pd.to_datetime(eps_df['report_period']).dt.tz_localize(None)
    merged = pd.merge_asof(
        daily_df.sort_values('Date'),
        eps_df.sort_values('report_period'),
        left_on='Date', right_on='report_period',
        direction='backward'
    )
    # Forward-fill any missing TTM_EPS values
    merged['TTM_EPS'] = merged['TTM_EPS'].ffill()
    merged['PE_Ratio'] = merged['Close'] / merged['TTM_EPS']
    merged.set_index('Date', inplace=True)
    return merged

# 4. Plot the daily P/E ratios for multiple tickers
def plot_pe_ratios(ticker_pe_dict):
    plt.figure(figsize=(12, 8))
    for ticker, df in ticker_pe_dict.items():
        if df is not None and 'PE_Ratio' in df.columns:
            plt.plot(df.index, df['PE_Ratio'], label=ticker)
    plt.xlabel('Date')
    plt.ylabel('P/E Ratio')
    plt.title('Historical Trailing Twelve Month P/E Ratio (using yfinance)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. Main function to process multiple tickers
def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Add additional tickers as needed
    ticker_pe_dict = {}
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        daily_prices = get_stock_data(ticker, start_date='2010-01-01')
        eps_df = get_quarterly_ttm_eps(ticker)
        if eps_df is None:
            print(f"Skipping {ticker} due to missing EPS data.")
            continue
        merged_df = align_eps_with_prices(daily_prices, eps_df)
        if merged_df is None:
            print(f"Skipping {ticker} due to error aligning EPS data.")
            continue
        ticker_pe_dict[ticker] = merged_df
        print(f"Latest P/E for {ticker}: {merged_df['PE_Ratio'].iloc[-1]}")
    
    plot_pe_ratios(ticker_pe_dict)

if __name__ == "__main__":
    main()