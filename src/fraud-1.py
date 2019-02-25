import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fancyimpute import KNN
import gmplot


def parse_data_for_model(df):
    # create column of fraud or not fraud
    df['fraud'] = 1 * df.acct_type.str.contains('fraud')
    # fill some columns with zero if NaN value
    df['has_header'].fillna(0, inplace = True)
    df['delivery_method'].fillna(0, inplace=True)
    df['org_facebook'].fillna(0, inplace=True)
    df['org_twitter'].fillna(0, inplace=True)
    # create column of whether the observation has a location or not
    df['has_location'] = 1 * df['venue_latitude'].notnull()
    df['listed'] = df['listed'].map({'y': 1, 'n': 0})
    # create dummy variables
    dummy1 = pd.get_dummies(df['currency'])
    dummy2 = pd.get_dummies(df['payout_type'])
    df = pd.concat([df, dummy1, dummy2], axis=1)
    # drop rows
    df.drop(['acct_type', 'approx_payout_date', 'channels', 'country', 'description',
    'email_domain', 'event_created', 'event_end', 'event_published', 'event_start',
    'gts', 'name', 'num_order', 'object_id', 'org_desc', 'org_name', 'payee_name',
    'previous_payouts', 'sale_duration2', 'ticket_types', 'user_created', 'venue_address',
    'venue_country', 'venue_name', 'venue_state', 'currency', 'USD', 'payout_type',
    'CHECK', ''], axis=1, inplace=True)
    cols = df.columns
    # use KNN to impute missing longitude and latitude data
    # using long, lat rather than other location features
    df = KNN(k=3).fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = cols
    return df


if __name__ == '__main__':
    # read in df
    df1 = pd.read_json('../data/data.zip')
    df1['fraud'] = 1 * df1.acct_type.str.contains('fraud')
    df2 = parse_data_for_model(df1)
    # df = df1[np.isfinite(df1['venue_latitude'])]
    # lats = df["venue_latitude"]
    # longs = df["venue_longitude"]
    # gmap = gmplot.GoogleMapPlotter(34.0522, -118.2437,1)
    # gmap.heatmap(lats, longs)
    # df2['fraud'] = 1 * df2.acct_type.str.contains('fraud')
'''
    fraud_df = df1[df1.fraud == 1]
    fraud_lats = fraud_df["venue_latitude"]
    fraud_longs = fraud_df["venue_longitude"]
    gmap2 = gmplot.GoogleMapPlotter(34.0522, -118.2437,1)
    gmap2.heatmap(fraud_lats, fraud_longs)

    not_fdf = df1[df1.fraud == 0]
'''
