import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fancyimpute import KNN


def format_data(df):
    # create column of fraud or not fraud
    if 'acct_type' in df.columns:
        df['fraud'] = 1 * df.acct_type.str.contains('fraud')

    # fill columns with zero if NaN value
    fill_cols = ['has_header', 'delivery_method', 'org_facebook', 'org_twitter']

    for col in fill_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    # df['has_header'].fillna(0, inplace = True)
    # df['delivery_method'].fillna(0, inplace=True)
    # df['org_facebook'].fillna(0, inplace=True)
    # df['org_twitter'].fillna(0, inplace=True)

    # create 'has_location' column
    if 'venue_latitude' in df.columns:
        df['has_location'] = 1 * df['venue_latitude'].notnull()

    # convert 'listed' column to binary
    if 'listed' in df.columns:
        df['listed'] = df['listed'].map({'y': 1, 'n': 0})

    # get dummy variables
    dummy1 = pd.get_dummies(df['currency'])
    dummy2 = pd.get_dummies(df['payout_type'])
    df = pd.concat([df, dummy1, dummy2], axis=1)

    df = drop_columns(df)
    df = impute_missing_data(df)

    return df

def impute_missing_data(df):
    cols = df.columns

    # use KNN to impute missing longitude and latitude data
    df = KNN(k=3).fit_transform(df)

    # reconstruct imputed df
    df = pd.DataFrame(df)
    df.columns = cols

    return df

# drop columns
def drop_columns(df):
    cols_to_drop = ['acct_type', 'approx_payout_date', 'channels', 'country', 'description',
    'email_domain', 'event_created', 'event_end', 'event_published', 'event_start',
    'gts', 'name', 'num_order', 'object_id', 'org_desc', 'org_name', 'payee_name',
    'previous_payouts', 'sale_duration2', 'ticket_types', 'user_created', 'venue_address',
    'venue_country', 'venue_name', 'venue_state', 'currency', 'USD', 'payout_type',
    'CHECK', '']

    dropped = [x for x in cols_to_drop if x in df.columns]

    df.drop(dropped, axis=1, inplace=True)

    return df


if __name__ == '__main__':
# #     # read in df
    df1 = pd.read_json('../data/data.zip')
    df1 = format_data(df1)
