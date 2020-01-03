"""A minimal example of using Gro Intelligence's API client library.

In this example, we retrieve Wheat Area Harvested for Ukraine from multiple data sources and print
the results to the console.
"""

import os
from api.client.gro_client import GroClient

API_HOST = 'api.gro-intelligence.com'

# Replace os.environ['GROAPI_TOKEN'] with "your-token" if you don't have your token saved to an
# environment variable. Environment variables are not required, but they are used throughout the
# example scripts.
# See https://developers.gro-intelligence.com/authentication.html#saving-your-token-as-an-environment-variable
# for more information.
ACCESS_TOKEN = os.environ['GROAPI_TOKEN'] 


def main():
    client = GroClient(API_HOST, ACCESS_TOKEN)

    # To find the IDs of entities, see https://developers.gro-intelligence.com/searching-data.html
    selected_entities = {'region_id': 1210,  # Ukraine
                         'item_id': 95,  # Wheat
                         'metric_id': 570001}  # Area Harvested (area)

    # Get what possible series there are for that combination of selections
    for data_series in client.get_data_series(**selected_entities):
        print('\nData series {}'.format(data_series))
        # Add a time range restriction to your data request
        # (Optional - otherwise get all points)
        data_series['start_date'] = '2000-01-01'
        data_series['end_date'] = '2012-12-31'

        for point in client.get_data_points(**data_series):
            print('{}: {} {}'.format(point['end_date'],
                                     point['value'],
                                     client.lookup_unit_abbreviation(point['unit_id'])))


if __name__ == "__main__":
    main()
