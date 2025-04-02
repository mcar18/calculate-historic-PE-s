import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup

# ========================
# 1. Get Daily Price Data from yfinance
# ========================
def get_stock_data(ticker, start_date='2010-01-01', end_date=None):
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df = df.reset_index()  # Ensure 'Date' is a column
    return df

# ========================
# 2. Download EDGAR Filings
# ========================
def download_filings(ticker, filing_type="10-Q", num_filings=8, download_folder="sec_filings", email="mattcarlson39@gmail.com"):
    # Initialize the downloader with your email
    dl = Downloader(download_folder, email)
    # Call get() without the extra parameter.
    dl.get(filing_type, ticker)
    ticker_folder = os.path.join(download_folder, ticker, filing_type)
    if not os.path.exists(ticker_folder):
        print(f"No filings found for {ticker} for {filing_type}.")
        return []
    filings = sorted(os.listdir(ticker_folder))
    # Limit to the most recent num_filings if available
    if len(filings) > num_filings:
        filings = filings[-num_filings:]
    filing_paths = [os.path.join(ticker_folder, filing) for filing in filings]
    return filing_paths

# ========================
# 3. Parse XBRL Instance Document for Net Income and Report Date
# ========================
def parse_xbrl_for_net_income(filing_folder):
    # Look for an instance document (XML file) in the filing folder
    instance_file = None
    for file in os.listdir(filing_folder):
        if file.endswith(".xml"):
            instance_file = os.path.join(filing_folder, file)
            break
    if instance_file is None:
        print(f"No instance XML file found in {filing_folder}.")
        return None, None

    with open(instance_file, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'lxml')
    # Look for a tag whose name ends with 'NetIncomeLoss' (common in US GAAP filings)
    net_income_tag = soup.find(lambda tag: tag.name.lower().endswith("netincomeloss"))
    if net_income_tag and net_income_tag.text:
        try:
            net_income = float(net_income_tag.text.strip().replace(',', ''))
        except Exception as e:
            print(f"Error parsing net income: {e}")
            net_income = None
    else:
        net_income = None

    # Extract report period from the first context's endDate, if available
    context = soup.find("xbrli:context")
    report_period = None
    if context:
        end_date_tag = context.find("xbrli:enddate")
        if end_date_tag:
            try:
                report_period = pd.to_datetime(end_date_tag.text.strip())
            except Exception as e:
                print(f"Error parsing report period: {e}")
                report_period = None

    return net_income, report_period

# ========================
# 4. Build Quarterly Financials DataFrame from EDGAR
# ========================
def build_quarterly_financials(ticker, filing_type="10-Q", num_filings=8, download_folder="sec_filings", email="mattcarlson39@gmail.com"):
    filing_paths = download_filings(ticker, filing_type, num_filings, download_folder, email)
    records = []
    for path in filing_paths:
        net_income, report_period = parse_xbrl_for_net_income(path)
        if net_income is not None and report_period is not None:
            records.append({"report_period": report_period, "NetIncome": net_income})
    if not records:
        print(f"No valid financial records found for {ticker} from EDGAR.")
        return None
    df = pd.DataFrame(records)
    df.sort_values("report_period", inplace=True)
    df.set_index("report_period", inplace=True)
    return df

# ========================
# 5. Calculate TTM EPS from EDGAR Quarterly Data
# ========================
def calculate_ttm_eps_edgar(quarterly_df, shares_outstanding):
    if quarterly_df is None or quarterly_df.empty:
        return None
    quarterly_df['TTM_NetIncome'] = quarterly_df['NetIncome'].rolling(window=4).sum()
    quarterly_df['TTM_EPS'] = quarterly_df['TTM_NetIncome'] / shares_outstanding
    ttm_df = quarterly_df.dropna(subset=['TTM_EPS'])
    return ttm_df[['TTM_EPS']]

# ========================
# 6. Align TTM EPS with Daily Price Data and Compute P/E
# ========================
def align_eps_with_prices(daily_df, eps_df):
    if eps_df is None or eps_df.empty:
        return None
    eps_df = eps_df.reset_index()
    daily_df = daily_df.copy()
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

# ========================
# 7. Plot Daily P/E Ratios for Multiple Tickers
# ========================
def plot_pe_ratios(ticker_pe_dict):
    plt.figure(figsize=(12, 8))
    for ticker, df in ticker_pe_dict.items():
        if df is not None and 'PE_Ratio' in df.columns:
            plt.plot(df.index, df['PE_Ratio'], label=ticker)
    plt.xlabel('Date')
    plt.ylabel('P/E Ratio')
    plt.title('Historical Trailing Twelve Month P/E Ratio (EDGAR + yfinance)')
    plt.legend()
    plt.grid(True)
    plt.show()

# ========================
# 8. Main Function: Process Multiple Tickers
# ========================
def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Add more tickers as needed
    email = "mattcarlson39@gmail.com"  # Replace with your actual email for EDGAR requests
    download_folder = "sec_filings"
    ticker_pe_dict = {}

    for ticker in tickers:
        print(f"\nProcessing {ticker} with EDGAR fundamentals...")
        daily_prices = get_stock_data(ticker, start_date='2010-01-01')
        quarterly_financials = build_quarterly_financials(ticker, filing_type="10-Q", num_filings=8, download_folder=download_folder, email=email)
        if quarterly_financials is None:
            print(f"Skipping {ticker} due to missing EDGAR financial data.")
            continue
        info = yf.Ticker(ticker).info
        shares = info.get("sharesOutstanding", None)
        if shares is None:
            print(f"Skipping {ticker} due to missing shares outstanding info.")
            continue
        ttm_eps_df = calculate_ttm_eps_edgar(quarterly_financials, shares)
        if ttm_eps_df is None or ttm_eps_df.empty:
            print(f"Skipping {ticker} due to insufficient EDGAR data for TTM EPS.")
            continue
        merged_df = align_eps_with_prices(daily_prices, ttm_eps_df)
        if merged_df is None:
            print(f"Skipping {ticker} due to error aligning EDGAR EPS with price data.")
            continue
        ticker_pe_dict[ticker] = merged_df
        print(f"Latest P/E for {ticker}: {merged_df['PE_Ratio'].iloc[-1]}")

    plot_pe_ratios(ticker_pe_dict)

if __name__ == "__main__":
    main()