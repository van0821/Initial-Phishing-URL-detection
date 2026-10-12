import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

air_quality = pd.read_csv("air_quality_no2.csv",index_col = 0,parse_dates = True)
#print(air_quality.head())

# air_quality.plot()
# plt.show()

# air_quality["station_paris"].plot()
# plt.show()

# air_quality.plot.scatter(x="station_london", y="station_paris", alpha = 0.5)
# plt.show()

# print(
# [
#     method_name
#     for method_name in dir(air_quality.plot)
#     if not method_name.startswith("_")
# ]
# )

# air_quality.plot.box()
# plt.show()

# axs = air_quality.plot.area(figsize=(12,4), subplots=True)
# plt.show()

fig, axs = plt.subplots(figsize = (12, 4))
air_quality.plot.area(ax = axs)
axs.set_ylabel("NO2_concentration")
fig.savefig("No2_concentration.png")
plt.show()

