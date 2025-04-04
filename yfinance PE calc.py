import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
#test
# 1. Get daily stock price data (adjusted for splits) using yfinance's "max" period.
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period='max')  # Get all available daily data
    df = df.reset_index()  # Ensure 'Date' is a column
    return df

# 2. Get quarterly financials from yfinance and compute TTM EPS.
#    This function looks for "Net Income Common Stockholders" first, then "Net Income"
def get_quarterly_ttm_eps(ticker):
    t = yf.Ticker(ticker)
    q_fin = t.quarterly_financials
    if q_fin.empty:
        print(f"Quarterly financials for {ticker} is empty.")
        return None
    print(f"\nQuarterly financials index for {ticker}: {list(q_fin.index)}")
    possible_rows = ["Net Income Common Stockholders", "Net Income"]
    net_income_key = None
    for candidate in possible_rows:
        for idx in q_fin.index:
            if candidate.lower() == idx.lower():
                net_income_key = idx
                break
        if net_income_key:
            break
    if not net_income_key:
        print(f"No suitable 'Net Income' row found for {ticker}. Available rows: {list(q_fin.index)}")
        return None

    net_income = q_fin.loc[net_income_key].sort_index()  # Ensure dates ascending
    # Rolling sum of 4 quarters => TTM net income
    ttm_net_income = net_income.rolling(window=4).sum()
    
    # Use current sharesOutstanding (yfinance does not provide historical shares)
    info = t.info
    shares = info.get("sharesOutstanding", None)
    if shares is None:
        print(f"Shares outstanding not found for {ticker}.")
        return None
    ttm_eps = ttm_net_income / shares
    df = ttm_eps.reset_index()
    df.columns = ["report_period", "TTM_EPS"]
    df.sort_values("report_period", inplace=True)
    return df

# 3. Merge the TTM EPS series with daily price data using merge_asof.
def align_eps_with_prices(daily_df, eps_df):
    if eps_df is None or eps_df.empty:
        return None
    eps_df = eps_df.copy()
    daily_df = daily_df.copy()
    # Convert date columns to timezone-naive datetimes
    daily_df['Date'] = pd.to_datetime(daily_df['Date']).dt.tz_localize(None)
    eps_df['report_period'] = pd.to_datetime(eps_df['report_period']).dt.tz_localize(None)
    merged = pd.merge_asof(
        daily_df.sort_values('Date'),
        eps_df.sort_values('report_period'),
        left_on='Date', right_on='report_period',
        direction='backward'
    )
    merged['TTM_EPS'] = merged['TTM_EPS'].ffill()
    merged['PE_Ratio'] = merged['Close'] / merged['TTM_EPS']
    merged.set_index('Date', inplace=True)
    return merged

# 4. Plot the daily P/E ratios for multiple tickers.
def plot_pe_ratios(ticker_pe_dict):
    plt.figure(figsize=(12, 8))
    for ticker, df in ticker_pe_dict.items():
        if df is not None and 'PE_Ratio' in df.columns:
            plt.plot(df.index, df['PE_Ratio'], label=ticker)
    plt.xlabel('Date')
    plt.ylabel('P/E Ratio')
    plt.title('Historical Trailing Twelve Month P/E Ratio (yfinance only)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. Main function to process multiple tickers.
def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Adjust tickers as needed
    ticker_pe_dict = {}
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        daily_prices = get_stock_data(ticker)
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