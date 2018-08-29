#!/usr/bin/env python2

import bs4
import matplotlib.pyplot as plt

with open("wikipedia.html") as h:
	soup = bs4.BeautifulSoup(h.read(), "lxml")

data = []

tables = soup.find_all("table", {"class" : "wikitable"})
for table in tables: 
	headers = map(lambda x: x.text.strip(), table.find_all("th"))
	if "Datum" in headers and "Einwohner" in headers:
		for row in table.find_all("tr"):

			def sanitize(text):
				text = text.strip()
				return text

			columns = list(map(lambda x: sanitize(x.text), row.find_all("td")))
			
			if columns == [] or not "31. Dezember" in columns[0]: continue

			year = int(columns[0].split(" ")[-1])
			citizens = int(columns[1].replace(".", ""))
			data += [[year, citizens]]

years = list(map(lambda x: x[0], data))
citizens = list(map(lambda x: x[1], data))

print (years)

# Calculate w1
N = len(years)
w1 = (N * sum([years[j] * citizens[j] for j in range(N)]) - sum([years[j] for j in range(N)]) * sum([citizens[j] for j in range(N)])) / (N * sum([years[j]**2 for j in range(N)]) - sum([years[j] for j in range(N)])**2)

w0 = (sum([citizens[j] for j in range(N)]) - w1 * sum([years[j] for j in range(N)])) / N
print (w0, w1)

# Plot the graph
print ([1888, w0+1888*w1], [2016, w0 + 2016 * w1])

plt.gcf().subplots_adjust(left=0.15)

plt.plot([1888, 2016], [w0+1888*w1, w0 + 2016 * w1], "k-", label="Linear Regressor")
plt.scatter(years, citizens, marker="x", label="Dataset")
plt.xlabel("Year")
plt.ylabel("Population of Aachen")
plt.legend(loc="best")

plt.savefig('plot.png', figsize=(8, 6), dpi=400,)
plt.savefig('plot.pgf')