from surprise import Dataset, Reader
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import pickle
from collections import Counter
import gradio as gr
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def read():
    reviews = pd.read_csv("reviews_cleaned.csv")
    reviewed_courses = reviews["course_id"].unique()

    courses = pd.read_csv("Course_info.csv")
    courses = courses[["id", "title", "category", "course_url"]]
    courses = courses[courses["id"].isin(reviewed_courses)]

    reader = Reader(rating_scale=(0, 5))
    surprise_data = Dataset.load_from_df(reviews[["user_id", "course_id", "rate"]], reader)
    trainset = surprise_data.build_full_trainset()
    return reviews, reviewed_courses, courses, trainset

algo = pickle.load(open("surprise.sav", 'rb'))

def random_user(df, trainset):
    users = df['user_id'].unique()
    index = np.random.choice(users.shape[0], 1, replace=False)
    user_id = users[index][0]
    return user_id, trainset.to_inner_uid(user_id)

def get_recommendations_by_category(user, df, courses, trainset, algo):
    predictions = list()
    user_courses_list = list(df[df["user_id"] == user[0]]["course_id"])
    user_courses = courses[courses["id"].isin(user_courses_list)]
    cnt = Counter(user_courses["category"].array)
    user_category = cnt.most_common(1)[0][0]
    category_courses = courses.loc[courses["category"] == str(user_category)]
    for course in category_courses.id:
        course_id = trainset.to_inner_iid(course)
        pred = algo.predict(user[1], course_id)
        predictions.append([course, pred.est])
    predictions = np.array(predictions)
    return predictions[predictions[:,1].argsort()][::-1][:3]

def get_recommendations(user, courses, trainset, algo):
    predictions = list()
    for course in courses.id:
        course_id = trainset.to_inner_iid(course)
        pred = algo.predict(user[1], course_id)
        predictions.append([pred.uid, pred.iid, pred.r_ui, pred.est, pred.details, user[0], course])
        df = pd.DataFrame(np.array(predictions), columns=['uid', 'iid', 'rui', 'est', 'details', "user_id", "course_id"])
    return df.sort_values(by='est')[:5]

def get_df_category():
    reviews, reviewed_courses, courses, trainset = read()
    user = random_user(reviews, trainset)
    user_courses = list(reviews[reviews["user_id"] == user[0]]["course_id"])
    recom = get_recommendations_by_category(user, reviews, courses, trainset, algo)
    recom_id = recom[:,0]
    return courses[courses["id"].isin(user_courses)], courses[courses["id"].isin(recom_id)], user[0]

def get_df_recom():
    reviews, reviewed_courses, courses, trainset = read()
    user = random_user(reviews, trainset)
    user_courses = list(reviews[reviews["user_id"] == user[0]]["course_id"])
    recom = get_recommendations(user, courses, trainset, algo)
    df = courses[courses["id"].isin(recom["course_id"].to_numpy())]
    return courses[courses["id"].isin(user_courses)], df, user[0]

def select(chk):
    return get_df_category() if chk else get_df_recom()

def get_course_recom_df():
    reviews, reviewed_courses, courses, trainset = read()
    table = pd.pivot_table(reviews, values="rate", columns="user_id", index="course_id").fillna(0)
    course = np.random.choice(table.shape[0])
    matrix = csr_matrix(table.values)
    knn = NearestNeighbors(metric = "cosine")
    knn.fit(matrix)
    dist, ind = knn.kneighbors(table.iloc[course].values.reshape(1, -1), n_neighbors=6)
    df1 = pd.DataFrame(courses[courses["id"] == table.index[ind.flatten()[0]]], columns=["id", "title", "category", "course_url"])
    df2 = pd.DataFrame(courses[courses["id"].isin(table.index[ind.flatten()[1:]])], columns=["id", "title", "category", "course_url"])
    return df1, df2, table.index[ind.flatten()[0]]
        
with gr.Blocks() as user:
    gr.Markdown("# AIN311 Project P05 - MOOC Recommendation")
    btn1 = gr.Button("Select a random user and recommend courses")
    chk = gr.Checkbox(label="Make recommendation according to user's category preference?")
    with gr.Column():
        tb1 = gr.Textbox(label="USER ID", interactive=False)
        d1 = gr.DataFrame(interactive=False, label="Courses Taken by the User", col_count=4, max_cols=4)
        d2 = gr.DataFrame(interactive=False, label="Recommended Courses", col_count=4, max_cols=4)        
    btn1.click(fn=select,inputs=[chk], outputs=[d1, d2, tb1])
    
with gr.Blocks() as course:
    gr.Markdown("# AIN311 Project P05 - MOOC Recommendation")
    btn2 = gr.Button("Select a random course and make recommendation based on that course")
    with gr.Column():
        tb2 = gr.Textbox(label="COURSE ID", interactive=False)
        d3 = gr.DataFrame(interactive=False, label="Course", col_count=4, max_cols=4) 
        d4 = gr.DataFrame(interactive=False, label="Recommended Courses", col_count=4, max_cols=4) 
    btn2.click(fn=get_course_recom_df, inputs=[], outputs=[d3,d4,tb2])
    
with gr.Blocks() as rate:
    gr.Markdown("# AIN311 Project P05 - MOOC Recommendation")

demo = gr.TabbedInterface([user, course, rate], ["By User", "By Course", "Rating Prediction"])
demo.launch()