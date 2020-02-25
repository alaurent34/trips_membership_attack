"""
File: membership_attack.py
Author: Antoine Laurent
Email: laurent.antoine@courrier.uqam.ca
Github: https://github.com/alaurent34
Description: Framework to perform membership attack on trips dataset
"""

import os
from itertools import combinations
from random import sample

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score,\
                            recall_score, confusion_matrix,\
                            accuracy_score, roc_curve, auc

from numba import jit, vectorize, int32, int64, float32, float64
from numba.typed import List

# ------------ VARIABLES DEFINITION --------------- #

# gather data
TRIPS_ID_TARGETTED = pd.read_csv("./data/trips_id_target.csv")
USERS_TARGETTED = np.sort(
    np.load("./data/uuid_target.npy", allow_pickle=1)
)
TRIPS = pd.read_csv("./data/trips.csv")

# set group size return by the oracle
GROUP_SIZES = [100, 1500, 3000]

# set testing and training size
TR_SIZE = 0.25                       # 1 is all the traj, 0 is none of them
TS_SIZE = 0.75                       # 1 is all the traj, 0 is none of them

# set the number of instance generated by Obs (ts) and Adv (tr)
TR_DATA_SIZE = 10
TS_DATA_SIZE = 100

# set the number of points and trips in the Adv knowledge
TR_POINTS_NB = 4
TR_TRIPS_NB = 2

# set the rounding value of the geospatial points
ROUNDING = 3

# set the trips columns
TRIPS_COLUMNS = ['trip_id', 'lat_first', 'lat_last', 'lng_first', 'lng_last', 'uuid']

# set the result path
LACIE = True
LACIE_PATH = "/run/media/antoine/LaCie/"
TR_TS_NAME = f"tr_size={TR_SIZE}_ts_size={TS_SIZE}"
INSTANCE_NAME = TR_TS_NAME + f"_nb_adv_trips={TR_TRIPS_NB}_nb_adv_points={TR_POINTS_NB}"
RESULT_PATH = f"./results/{INSTANCE_NAME}/"
os.makedirs(RESULT_PATH, exist_ok=True)

# ------------ END VARIABLES DEFINITION --------------- #

# sample a number of cabs NOT including the target cab, and return the list of cab ids
#@jit(forceobj=True)
def sample_uuids_no_target(uuids, target_uuid, group_size):
    """
    doc
    """

    # make sure the group size cannot be larger
    assert group_size < len(uuids)

    # create a set out of the target cab
    set_1 = set([target_uuid])

    # create a set out of the rest of cabs and convert it to list
    set_2 = list(set(uuids) - set_1)

    # sample the cabs list not containing the target
    set_3 = sample(set_2, group_size)

    return sorted(list(set_3))

# sample a number of uuid including the target uuid, and return the list of uuid
#@jit(forceobj=True)
def sample_uuids_with_target(uuids, target_uuid, group_size):
    """
    doc
    """

    assert group_size <= len(uuids)

    # create a set out of the target cab list
    set_1 = set([target_uuid])

    # create a set out of the rest of cabs and convert it to list
    set_2 = list(set(uuids) - set_1)

    # sample the cabs list not containing the target
    set_3 = sample(set_2, group_size - 1)

    # get the union of the sampled set and the target cab
    set_4 = set(set_3) | set_1

    return sorted(list(set_4))

# return a list of lists where each item is a group
#@jit(forceobj=True)
def sample_unique_groups(trips, target_uuid, group_size, ts_data_size):
    """
    doc
    """

    groups_in, groups_out = [], []
    uuids = trips[trips.uuid != target_uuid].uuid

    # fill the list with groups containing the user
    while len(groups_in) < (ts_data_size / 2.0):

        group = sample_uuids_with_target(uuids, target_uuid, group_size)
        groups_in.append(group)

    # fill the list with groups not containing the user
    while len(groups_out) < (ts_data_size / 2.0):

        group = sample_uuids_no_target(uuids, target_uuid, group_size)
        groups_out.append(group)

    return groups_in, groups_out

#@jit(forceobj=True)
def trip_id_from_uuids(trips, uuids, target_uuid, ts_trips_id):
    """
    doc
    """

    if (np.array(uuids) != target_uuid).all():
        trips_ids = np.unique(trips[trips.uuid.isin(uuids)].trip_id.values)
    else:
        # remove uuid from trips, extract all trips and then add ts_trips_id reprensenting testing
        trips_ids = trips[trips.uuid != target_uuid]
        trips_ids = np.unique(trips[trips.uuid.isin(uuids)].trip_id.values)
        trips_ids = np.concatenate([trips_ids, ts_trips_id], axis=None)

    return trips_ids

# @jit(forceobj=True)
def sample_data(data, nb_repeat=4, nb_instance=10):
    """
    Generate nb_instance of nb_data data to simulate the auxiliary knowledge of Adv.
    Data is trips or points.
    """

    np.random.seed(42)                            # repeatability
    # create the cartesian product of nb_repeat
    comb = combinations(data, nb_repeat)

    comb = np.array([x for x in comb])
    np.random.shuffle(comb)

    # # remove all products that are composed of two identical data
    # all_different = List()
    # for i in range(cartesian.shape[0]):
    #     shape_instance = np.unique(cartesian[i], axis=0).shape[0]
    #     if shape_instance == nb_repeat:
    #         all_different.append(cartesian[i])

    # # shuffle data
    # all_different = np.array(all_different)
    # np.random.shuffle(all_different)
    if comb.shape[0] <= nb_instance:
        nb_instance = comb.shape[0]


    return comb[:nb_instance]

#@jit(forceobj=True)
def fetch_user_trips(trips, tr_trips_id, rounding=ROUNDING, nb_instance=10):
    """
    doc
    """

    # Recover all trips from user uuid
    tr_trips = trips[trips.trip_id.isin(tr_trips_id)]
    tr_trips.columns = TRIPS_COLUMNS

    # recover trips only
    tr_trips = np.array(tr_trips[["lat_first", "lng_first", "lat_last", "lng_last"]])

    # round point to rounding value : default is 2 (800m precision)
    tr_trips = np.round(tr_trips, rounding)
    tr_trips = np.unique(tr_trips, axis=0)       # remove duplicates

    # create the prior knowledge of Adv
    tr_data = sample_data(tr_trips, nb_repeat=TR_TRIPS_NB, nb_instance=nb_instance)

    return tr_data

#@jit(forceobj=True)
def fetch_user_points(trips, tr_trips_id, rounding=ROUNDING, nb_instance=10):
    """
    doc
    """

    # Recover all points from user uuid
    tr_trips = trips[trips.trip_id.isin(tr_trips_id)]
    tr_trips.columns = TRIPS_COLUMNS
    first_points = np.array(tr_trips[["lat_first", "lng_first"]])
    last_points = np.array(tr_trips[["lat_last", "lng_last"]])
    points = np.concatenate([first_points, last_points])

    # round points to rounding value : default is 2 (800m precision)
    points = np.round(points, rounding)
    points = np.unique(points, axis=0)          # remove duplicates

    tr_data = sample_data(points, nb_repeat= TR_POINTS_NB, nb_instance=nb_instance)

    return tr_data

def user_data(trips, uuid, tr_trips_id, ts_trips_id, group_size,
              tr_data_size, ts_data_size, user_path):
    """
    doc
    """

    trips = trips.copy()
    trips.columns = TRIPS_COLUMNS

    # generate points and trips dataset representing Adv aux knowledge
    tr_data_points = fetch_user_points(trips, tr_trips_id, nb_instance=tr_data_size)
    tr_data_trips = fetch_user_trips(trips, tr_trips_id, nb_instance=tr_data_size)

    # generate batch of Obs challenges
    group_in, group_out = sample_unique_groups(trips, uuid, group_size, ts_data_size)

    # empty dataframe for the Obs values
    ts_data = pd.DataFrame(columns=TRIPS_COLUMNS+["group_id", "in"])

    # filling ts_data with trips of uuids in groups in
    for i, uuids in enumerate(group_in):
        trips_id_in = trip_id_from_uuids(trips, uuids, uuid, ts_trips_id)
        trips_grp = trips[trips.trip_id.isin(trips_id_in)].copy().reset_index(drop=True)
        trips_grp.loc[:, "group_id"] = i
        trips_grp.loc[:, "in"] = 0              # 0 means that the trips of the user are in

        ts_data = pd.concat([ts_data, trips_grp])

    # filling ts_data with trips of uuids in groups out
    for i, uuids in enumerate(group_out):
        trips_id_out = trip_id_from_uuids(trips, uuids, uuid, ts_trips_id)
        trips_grp = trips[trips.trip_id.isin(trips_id_out)].copy().reset_index(drop=True)
        trips_grp.loc[:, "group_id"] = len(group_in) + i
        trips_grp.loc[:, "in"] = 1              # 1 means that the trips of the user are out

        ts_data = pd.concat([ts_data, trips_grp])

    # saving data for reuse
    os.makedirs(user_path, exist_ok=1)
    np.save(user_path+"/tr_data_points", tr_data_points)
    np.save(user_path+"/tr_data_trips", tr_data_trips)
    ts_data.to_csv(user_path+"/ts_data.csv", index=False)

    return tr_data_points, tr_data_trips, ts_data.reset_index(drop=True)

@vectorize([int32(int32),
            int64(int64),
            float32(float32),
            float64(float64)])
def proba(x):
    """
    doc
    """
    if x == 0:
        return 0.0
    return 2**(-np.log(x))

def choose_class(x):
    """
    doc
    """
    if x <= 0.2:
        return 1
    return 0

@jit(nopython=True)
def prediction_with_count(occurrence_count):
    """
    doc
    """

    return proba(occurrence_count).sum()/occurrence_count.shape[0]

def points_attack(points, all_points):
    """
    doc
    """

    # for each point, count it's occurance in the all the trips
    count = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        count[i] = np.count_nonzero(np.isin(all_points, points[i]).all(axis=1))

    pred = prediction_with_count(count)

    return pred

@jit(nopython=True)
def trips_attack(trips, all_trips):
    """
    doc
    """

    # for each point, count it's occurance in the all the trips
    count = np.zeros(trips.shape[0])
    for i in range(trips.shape[0]):
        for j in range(all_trips.shape[0]):
            if (all_trips[j] == trips[i]).all():
                count[i] += 1

    pred = prediction_with_count(count)

    return pred

def predict(name, ts_data, tr_data, nb_instance_adv, nb_instance_chl, rounding=ROUNDING):
    """
    doc
    """

    # update nb_instance_adv (if there was not enought data to build nb_instance_adv in the first place)
    nb_instance_adv = min(nb_instance_adv, tr_data.shape[0])
    # update nb_instance_chl (if there was not enought data to build nb_instance_chl in the first place)
    updated_instance_chl = ts_data[["group_id", "in"]].drop_duplicates().set_index("group_id").shape[0]
    nb_instance_chl = min(nb_instance_chl, ts_data.shape[0])

    result = np.zeros((nb_instance_adv, nb_instance_chl))

    # for each auxiliare knowledge
    for i in range(nb_instance_adv):
        for j in range(nb_instance_chl):
            instance_adv = tr_data[i]
            instance_chl = ts_data[ts_data.group_id == j].copy()

            if name == "Points":
                # transform all_trips into a list of points
                all_points = np.concatenate(
                    [np.array(instance_chl[["lat_first", "lng_first"]]),
                     np.array(instance_chl[["lat_last", "lng_last"]])]
                )
                all_points = np.round(all_points, rounding)

                result[i, j] = points_attack(instance_adv, all_points)
            else:
                # transform all_trips into a list of points
                all_trips = np.array(
                    instance_chl[["lat_first", "lng_first", "lat_last", "lng_last"]]
                )
                all_trips = np.round(all_trips, rounding)
                result[i, j] = trips_attack(instance_adv, all_trips)

    return result

def scores(attack_type, uuid, group_size, labels, predictions, _scores):
    """
    doc
    """

    # create a new df to store results
    df_res_cols = ['uuid', 'attack', 'tp', 'fp', 'fn', 'tn', 'acc', 'ppv', 'tpr', 'fpr', 'auc', 'f1']

    # if there is a file for the users
    if os.path.isfile(RESULT_PATH + 'res_' + str(group_size) + '.csv'):
        res = pd.read_csv(RESULT_PATH + 'res_' + str(group_size) + '.csv')
    else:
        res = pd.DataFrame(columns=df_res_cols)

    conf = 0
    acc = 0
    ppv = 0
    rec = 0
    tnr = 0
    f1 = 0
    area = 0

    for i in range(predictions.shape[0]):
        conf += confusion_matrix(labels, predictions[i], labels=[0.0, 1.0])
        acc += accuracy_score(labels, predictions[i])
        ppv += precision_score(labels, predictions[i], pos_label=0.0, zero_division=0)
        rec += recall_score(labels, predictions[i], pos_label=0.0)
        tnr += recall_score(labels, predictions[i], pos_label=1.0)
        f1 += f1_score(labels, predictions[i], pos_label=0.0)

        # calculate fpr, tpr and auc for ROC curve plot
        fpr, tpr, _ = roc_curve(labels, _scores[i], pos_label=0)
        area += auc(fpr, tpr)

    conf //= predictions.shape[0]
    acc /= predictions.shape[0]
    ppv /= predictions.shape[0]
    rec /= predictions.shape[0]
    tnr /= predictions.shape[0]
    f1 /= predictions.shape[0]
    area /= predictions.shape[0]

    print("Mean Accuracy :", acc)
    # dataframe to store the results of the target user
    res = res.append(
        pd.Series([
            uuid, attack_type, conf[0][0], conf[1][0],
            conf[0][1], conf[1][1], acc, ppv, rec,
            1.0 - tnr, area, f1], index=df_res_cols),
        ignore_index=True
    )

    # save the new pickle to disk
    res.to_csv(RESULT_PATH + 'res_' + str(group_size) + '.csv', index=False)

def attack(trips, target, group_size, tr_trips_id, ts_trips_id,
           tr_data_size, ts_data_size, user_path):
    """
    doc
    """
    # read aux knowledge
    try:
        tr_data_points = np.load(user_path+"/tr_data_points.npy", allow_pickle=1)
        tr_data_trips = np.load(user_path+"/tr_data_trips.npy", allow_pickle=1)
        ts_data = pd.read_csv(user_path+"/ts_data.csv")
    except FileNotFoundError:
        print("Sampling Data...")
        tr_data_points, tr_data_trips, ts_data = user_data(trips,
                                                           target,
                                                           tr_trips_id,
                                                           ts_trips_id,
                                                           group_size,
                                                           tr_data_size,
                                                           ts_data_size,
                                                           user_path)

    attack_type = ["Points", "Trips"]

    # retrieve the group number with the label associated
    df_group_label = ts_data[["group_id", "in"]].drop_duplicates().set_index("group_id")
    df_group_label.columns = ["y_label"]
    df_group_label.sort_index()
    y_label = np.array(df_group_label["y_label"]).astype(int)

    datas = [tr_data_points, tr_data_trips]

    # perform label prediction for each groups and each attack : ie point and traj
    for name, data in zip(attack_type, datas):

        # probability scores
        _scores = predict(name, ts_data, data, tr_data_size, ts_data_size)

        # predict based on probability scores
        vchoose_class = np.vectorize(choose_class)
        y_preds = vchoose_class(_scores)

        #compute score
        scores(name, target, group_size, y_label, y_preds, _scores)

    return 0

def main(trips_id_targetted, users_targetted, trips, group_sizes,
         tr_size, ts_size, tr_data_size, ts_data_size):
    """
    TODO: Docstring for main.

    """
    for target in users_targetted:
        print("Target:", target)

        trips_user = trips_id_targetted[trips_id_targetted.uuid == target].trip_id.values

        # sample the training trip
        np.random.seed(seed=42) # for repeatability
        tr_trips_id = trips_user[np.random.choice(trips_user.shape[0],
                                                  int(trips_user.shape[0]*tr_size),
                                                  replace=False
                                                  )
                                 ]

        # sample the training trip
        if ts_size == tr_size and tr_size == 1:
            ts_trips_id = tr_trips_id
        else:
            ts_trips_id = trips_user[~np.isin(trips_user, tr_trips_id)]

        for group_size in group_sizes:
            print("Group Size:", group_size)
            # path precomputed data
            user_path = f"data/user-dfs/{INSTANCE_NAME}/user-{target}_groupsize-{group_size}_rounding-{ROUNDING}/"
            if LACIE:
                user_path = LACIE_PATH + "/membership/" + user_path

            # attack
            attack(
                trips, target, group_size,
                tr_trips_id, ts_trips_id,
                tr_data_size, ts_data_size, user_path
            )

if __name__ == "__main__":

    # launch attack
    main(
        TRIPS_ID_TARGETTED, USERS_TARGETTED, TRIPS,
        GROUP_SIZES, TR_SIZE, TS_SIZE, TR_DATA_SIZE,
        TS_DATA_SIZE
    )
