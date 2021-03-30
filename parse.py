
import urllib.request as req
import urllib.error as er
import re


def openFile(f):
    try:
        return req.urlopen(f)
    except er.HTTPError as httpe:
        print(httpe)
    except er.URLError as urle:
        print(urle)


def extractDates(soup_handle, today_string):
    dates = []
    day_node = -1
    node_count = -1
    date_html = soup_handle.find_all('div', {'class': 'obs-date'})
    for d_node in date_html:
        for c in d_node.children:
            node_count += 1
            date = c.text.strip()
            # start with today
            if len(re.findall(today_string, date)) > 0:
                day_node = node_count
            if day_node > -1:
                dates.append(date)
    return dates, day_node


def extractForecasts(soup_handle, day_node):
    highs = []
    lows = []
    forecasts = []
    node_count = -1
    forecast_html = soup_handle.find_all('div', {'class': 'obs-forecast'})
    for f_node in forecast_html:
        node_count += 1
        if node_count >= day_node:
            d_counter = -1
            for d in f_node.descendants:
                d_counter += 1
                if d_counter == 1: highs.append(d.string.strip())
                if d_counter == 5: lows.append(d.string.strip())
                if d_counter == 11: forecasts.append(d.string.strip())
    return highs, lows, forecasts


def extractPrecipitation(soup_handle, day_node):
    precip_types = []
    precip_inches = []
    node_count = -1
    precip_html = soup_handle.find_all('div', {'class': 'obs-precip'})
    for p_node in precip_html:
        node_count += 1
        # print("Precipitation Node: ", p_node)
        # print("\n")
        if node_count >= day_node:
            d_counter = -1
            for d in p_node.descendants:
                d_counter += 1
                # print("Node descendant: ", d)
                if d_counter == 4:
                    if d.string.strip() == 'Precip':
                        precip_types.append('Rain')
                    else:
                        precip_types.append(d.string.strip())
                if d_counter == 9:
                    amount = re.sub(r"\sin", "", d.string.strip())
                    # print("The amount is ", amount)
                    if len(re.findall(amount, "--")) > 0:
                        precip_inches.append(0)
                    else:
                        precip_inches.append(amount)
    return precip_types, precip_inches


def writeWidget(dates, highs, lows, forecasts, precip_types, precip_inches, num_days=3):
    print("<html>")
    print("<title>ED Cardiac Weather Watch</title>")
    print("<body>")
    if num_days == 3:
        print('<table width="25%">')
    elif num_days < 6:
        print('<table width="33%">')
    else:
        print('<table width="50%">')
    print('<tr><td align="center" colspan="', num_days, '"><b>')
    print("ED Cardiac Weather Widget")
    print('</b></td></tr><tr><td colspan="', num_days, '"></td></tr><tr>')
    for i in range(num_days):
        print("<td align='center'>", dates[i], "</td>")
    print("</tr><tr>")
    for i in range(num_days):
        print("<td align='center'>", highs[i], "/", lows[i], "</td>")
    print("</tr><tr>")
    for i in range(num_days):
        print("<td align='center'>", forecasts[i], "</td>")
    print("</tr><tr>")
    for i in range(num_days):
        if precip_inches[i] == '0':
            print("<td align='center'>No", precip_types[i], "</td>")
        else:
            print("<td align='center'>", precip_inches[i], "in", precip_types[i], "</td>")
    print("</tr>")
    snow_flag = False
    snow_day = ""
    if any(list(map(float, precip_inches))) >= 1.0:
        for i in range(num_days):
            inch = float(precip_inches[i])
            if precip_types[i] == 'Snow' and inch >= 1.0:
                snow_flag = True
                snow_day = dates[i]
    if snow_flag:
        print('<tr><td colspan="', num_days, '"></td></tr>')
        print("<tr>")
        print("<td align='center' colspan='", num_days, "'><b><font color='red'>1 or more in. SNOW coming", snow_day, "</font></b></td>")
        print("</tr>")
    print("</table></body></html>")