# calculate-historic-PE-s
gather financial data and use it to calculate PE ratios to evaluate a company

list of companies we want to evaluate. may expand later. tickers = ['SPY', 'AAPL', 'MSFT', 'MA', 'V', 'AXP', 'CRM', 'GOOG', 'NVDA', 'IBKR', 
#           'TSLA','AMD', 'KO','AMZN','META','GOOGL','AVGO','JPM','LLY','UNH',
#           'XOM','COST','NFLX','WMT','PG','JNJ','HD','ABBV','BAC','PM',
#           'CVX','CSCO','ABT','MCD','ORCL','IBM','WFC','PEP','MRK','GE']


```we aren't trading for the probabilty P that a stock will be within its confidence interval by that date, 
#we are predicting that probability P that the TRADE  will have moved in a positive direction from between entering and the conditions required to exit before the stock moves outside its confidence interval. 
And we should use certain exit conditions as preventative measures if a stock gets too close to that range.```