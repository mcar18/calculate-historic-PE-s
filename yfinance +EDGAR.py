import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
#we aren't trading for the probabilty P that a stock will be within its confidence interval by that date, 
#we are predicting that probability P that the TRADE will have moved in a positive direction from between entering and the conditions required to exit before the stock moves outside its confidence interval. 
#And we should use certain exit conditions as preventative measures if a stock gets too close to that range.  
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
# (email is the first parameter)
# ========================
def download_filings(email, ticker, filing_type="10-Q", num_filings=8, download_folder="sec-edgar-filings"):
    # Initialize the downloader with your email
    dl = Downloader(download_folder, email)
    # Download filings using filing_type and ticker (do not pass extra parameters)
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
# 3. Parse Filing Folder for Net Income and Report Date
# ========================
def parse_xbrl_for_net_income(filing_folder):
    # Try to find an instance document with a .xml extension first.
    instance_file = None
    for file in os.listdir(filing_folder):
        if file.endswith(".xml"):
            instance_file = os.path.join(filing_folder, file)
            break
    # If no .xml file is found, try to use full-submission.txt (case-insensitive)
    if instance_file is None:
        for file in os.listdir(filing_folder):
            if file.lower() == "full-submission.txt":
                instance_file = os.path.join(filing_folder, file)
                break
    if instance_file is None:
        print(f"No instance file found in {filing_folder}.")
        return None, None

    with open(instance_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Attempt to parse as XML first; if that fails, try as HTML.
    try:
        soup = BeautifulSoup(content, 'lxml')
    except Exception as e:
        print(f"Error parsing {instance_file} with lxml: {e}. Trying html.parser.")
        soup = BeautifulSoup(content, 'html.parser')

    # Look for a tag whose name ends with 'NetIncomeLoss' (case-insensitive)
    net_income_tag = soup.find(lambda tag: tag.name.lower().endswith("netincomeloss"))
    if net_income_tag and net_income_tag.text:
        try:
            net_income = float(net_income_tag.text.strip().replace(',', ''))
        except Exception as e:
            print(f"Error converting net income to float in {instance_file}: {e}")
            net_income = None
    else:
        net_income = None

    # Extract report period from the first context's endDate, if available.
    # Some filings may not have xbrli:context; if not, try to search for an alternative.
    context = soup.find("xbrli:context")
    report_period = None
    if context:
        end_date_tag = context.find("xbrli:enddate")
        if end_date_tag:
            try:
                report_period = pd.to_datetime(end_date_tag.text.strip())
            except Exception as e:
                print(f"Error parsing report period in {instance_file}: {e}")
                report_period = None
    else:
        # As a fallback, try to find a tag with "PeriodEnd" or similar
        possible_date_tag = soup.find(lambda tag: "periodend" in tag.text.lower())
        if possible_date_tag:
            try:
                report_period = pd.to_datetime(possible_date_tag.text.strip())
            except Exception as e:
                print(f"Error parsing fallback report period: {e}")
                report_period = None

    return net_income, report_period

# ========================
# 4b. Build Quarterly Financials DataFrame from EDGAR
# ========================
def build_quarterly_financials(email, ticker, filing_type="10-Q", num_filings=8, download_folder="sec-edgar-filings"):
    filing_paths = download_filings(email, ticker, filing_type, num_filings, download_folder)
    # If no filings found for 10-Q, try 10-K
    if not filing_paths:
        print(f"No filings found for {ticker} for {filing_type}. Trying 10-K instead.")
        filing_paths = download_filings(email, ticker, "10-K", num_filings, download_folder)
        if not filing_paths:
            print(f"No filings found for {ticker} for 10-K either.")
            return None
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
# 6b. Align TTM EPS with Daily Price Data and Compute P/E
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
    download_folder = "sec-edgar-filings"
    ticker_pe_dict = {}

    for ticker in tickers:
        print(f"\nProcessing {ticker} with EDGAR fundamentals...")
        daily_prices = get_stock_data(ticker, start_date='2010-01-01')
        quarterly_financials = build_quarterly_financials(email, ticker, filing_type="10-Q", num_filings=8, download_folder=download_folder)
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