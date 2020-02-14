"""
File: preprocessing.py
Author: Antoine Laurent
Email: laurent.antoine@courrier.uqam.ca
Github: https://github.com/alaurent34
Description: Preprocessing for data of membership attack
"""

import pandas as pd
import numpy as np

def sample_uuid(trips, nb_uuid, nb_trip_uuid=16, seed=1):
    """
    Sample a fraction `nb_uiid` (default 100) of users that have at least 16 trips in the data durng
    all the observation period.

    Parameters
    ----------
    trips : dataframe
        Database of coordinates that represent the trips
    nb_uuid : int
        Number of users targeted by Adv in the membership attack
    nb_trip_uuid : int
        Minimal number of trip per user in `trips`
    seed : int
        For reproductibility

    Returns
    -------
    trips_sampled : dataframe
        The list of sampled uuid along with their trip_id
    trips : dataframe
        Dataframe containing all first and last position of all users respecting `nb_trip_uuid`
    uuid_sampled : array
        List of all uuid sampled

    """

    trips = trips.copy()

    # Sample uuid on number of trips per uuid
    uuid_sample = np.sort(trips.uuid.unique())[
        trips.groupby(["uuid", "trip_id"]).first().groupby("uuid").size() >= nb_trip_uuid
    ]

    # get first and last point
    trips = trips[trips.uuid.isin(uuid_sample)]
    trips_first_last = trips.sort_values(["trip_id", "timestamp"])\
                            .groupby("trip_id")[["latitude", "longitude"]]\
                            .agg(["first", "last"])
    trips_first_last = pd.merge(
        trips_first_last,
        trips[["uuid", "trip_id"]].drop_duplicates(),
        on="trip_id"
    )

    # Get a sample of nb_uuid
    np.random.seed(seed=seed) # for repeatability
    uuid_sample = uuid_sample[np.random.choice(uuid_sample.shape[0], nb_uuid, replace=False)]

    # get the trips associated
    trips = trips[trips.uuid.isin(uuid_sample)]

    return trips[["uuid", "trip_id"]].drop_duplicates(), trips_first_last, uuid_sample

def membership_preprocessing():
    """
    Preprocess the data and save it
    """
    trips = pd.read_csv("../../data/preprocessed/csv/coo_alpha=100.csv")
    trips_sampled, trips, uuid_sampled = sample_uuid(trips, 100, 16, 1)
    trips.to_csv("../../data/preprocessed/membership/trips.csv", index=False)
    trips_sampled.to_csv(
        "../../data/preprocessed/membership/trips_id_target.csv",
        header="trip_id",
        index=False
    )
    np.save("../../data/preprocessed/membership/uuid_target", uuid_sampled)
