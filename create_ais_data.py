import csv

import pandas as pd

def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def process(chunk):
    shipIDs = chunk.MMSI
    shipIDindices = indices(shipIDs, shipIDs[1])

    latList = chunk.Latitude
    lonList = chunk.Longitude
    headingList = chunk.Heading

    f = open("shipdata.csv", "w")
    # print("Info for:", chunk.Name[0])
    for index, i in enumerate(shipIDindices):
        lat = latList[i]
        lon = lonList[i]
        heading = headingList[i]
        string = "{}, {}, {}, {}\n".format(index, lat, lon, heading)
        f.write(string)
    f.close()

chunksize = 10000000
for chunk in pd.read_csv("aisdk-2022-03-20.csv", chunksize=chunksize):
    process(chunk)
    break
