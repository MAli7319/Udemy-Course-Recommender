# Some of these methods directly taken from scikit-surprise documentation
# https://surprise.readthedocs.io/en/stable/FAQ.html

from collections import defaultdict
import random
from surprise import Dataset, KNNBasic

def get_top_n(predictions, n=3):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def get_anti_testset_user(full_train_set):
    anti_testset_user = []
    inner_id = random.choice(full_train_set.all_users())
    raw_id = full_train_set.to_raw_uid(inner_id)
    fill_value = full_train_set.global_mean
    user_item_ratings = full_train_set.ur[inner_id]
    user_items = [item for (item,_) in (user_item_ratings)]
    for iid in full_train_set.all_items():
        if iid not in user_items:
            anti_testset_user.append((raw_id,full_train_set.to_raw_iid(iid),fill_value))
    return anti_testset_user, raw_id

def get_nearest_neighbors(full_train_set, knn):
    inner_id = random.choice(full_train_set.all_items())
    raw_id = full_train_set.to_raw_iid(inner_id)
    similar_courses = knn.get_neighbors(inner_id, k=5)
    similar_courses_raw_ids = [full_train_set.to_raw_iid(i) for i in similar_courses]
    return similar_courses_raw_ids, raw_id