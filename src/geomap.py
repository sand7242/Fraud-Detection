import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fancyimpute import KNN
import gmplot
# from mpl_toolkits.basemap import Basemap
import pandas as pd
import io
import geopandas
from shapely.geometry import Point





if __name__ == '__main__':
    # read in df
    df1 = pd.read_json('../data/data.zip')
    # parse_data_for_model(df1)
    df = df1[np.isfinite(df1['venue_latitude'])]
    # lats = df["venue_latitude"]
    # longs = df["venue_longitude"]
    # gmap = gmplot.GoogleMapPlotter(34.0522, -118.2437,1)
    # gmap.heatmap(lats, longs)
    df1['fraud'] = 1 * df1.acct_type.str.contains('fraud')
    df1 = df1[np.isfinite(df1['venue_latitude'])]

    fraud_df = df1[df1.fraud == 1]
    # fraud_lats = fraud_df["venue_latitude"].values
    # fraud_longs = fraud_df["venue_longitude"].values
    fraud_df['Coordinates'] = list(zip(fraud_df["venue_longitude"], fraud_df["venue_latitude"]))
    fraud_df['Coordinates'] = fraud_df['Coordinates'].apply(Point)
    fraud_gdf = geopandas.GeoDataFrame(fraud_df, geometry='Coordinates')

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    #
    #
    # ax = world[world.continent != 'Antarctica'].plot(
    #     color='white', edgecolor='black')
    #
    # # We can now plot our GeoDataFrame.
    # fraud_gdf.plot(ax=ax, color='red', alpha=.3)
    # plt.axis('off')
    # plt.title('Cases of Fraud', fontsize=14)
    # plt.savefig('heatmap-fraud')






    nf_df = df1[df1.fraud == 0]
    # nf_lats = nf_df["venue_latitude"].values
    # nf_longs = nf_df["venue_longitude"].values
    nf_df['Coordinates'] = list(zip(nf_df["venue_longitude"], nf_df["venue_latitude"]))
    nf_df['Coordinates'] = nf_df['Coordinates'].apply(Point)
    nf_gdf = geopandas.GeoDataFrame(nf_df, geometry='Coordinates')

    ax = world[world.continent != 'Antarctica'].plot(
        color='white', edgecolor='black')

    # We can now plot our GeoDataFrame.
    nf_gdf.plot(ax=ax, color='red', alpha=.3)
    plt.axis('off')
    plt.title('Cases of Not Fraud', fontsize=14)
    # plt.show()
    plt.savefig('heatmap-not-fraud')
