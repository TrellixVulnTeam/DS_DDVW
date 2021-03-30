
import parse
import re
import datetime as dt
import bs4 as bs

dates = []
highs = []
lows = []
forecasts = []
precip_types = []
precip_inches = []
day_node = -1

# fake date to match one of the dates in the file
today_string = "1/15"
# what you would really do
# today = dt.date.today()
# today_string = today.strftime("%m/%d")

html = parse.openFile('file:///home/kylier/python/DS/data/weather.html')
soup_handle = bs.BeautifulSoup(html.read(), 'html.parser')
dates, day_node = parse.extractDates(soup_handle, today_string)
highs, lows, forecasts = parse.extractForecasts(soup_handle, day_node)
precip_types, precip_inches = parse.extractPrecipitation(soup_handle, day_node)

# default # of days to show is 3
# to show > 3 days, pass an extra argument with the # of days to show
parse.writeWidget(dates, highs, lows, forecasts, precip_types, precip_inches)


