import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import logging

def add_features(row):
    return row["keywords"] + " " + row["cast"] + " " + row["genres"]

#Use CountVectorizer to create a matrix holding similarity scores of different titles based on new row added before
#This will be put into a non-descriptive method create_rating_matrix()
def create_rating_matrix(movie_data):
    cv = CountVectorizer()
    rating_matrix = cv.fit_transform(movie_data["added_features"])
    cos_sim = cosine_similarity(rating_matrix)
    return cos_sim

#Create getters for titles and matrix indices to make retrieval simpler
def get_title(movie_data, index):
    return movie_data[movie_data.index == index]["title"].values[0]
def get_index(movie_data, title):
    return movie_data[movie_data.title == title]["index"].values[0]

#Create second non-descriptive method to do a fuzzy search of movie titles and find best recommendations
def get_recc_titles(movie_data):
    query = input("Enter a movie title: ")
    title_search = process.extractOne(query, movie_data["title"])
    confirm_title = input(f"The title that best matches your search was {title_search[0]}. Is this correct? (\'y\' if correct or press enter to search again): ")
    movie_title = title_search[0]

    while confirm_title != "y":
        query = input("Enter a movie title: ")
        title_search = process.extractOne(query, movie_data["title"])
        confirm_title = input(f"The title that best matches your search was {title_search[0]}. Is this correct? (\'y\' if correct or press enter to search again): ")
        movie_title = title_search[0]

    movie_index = get_index(movie_data, movie_title)

    # create a list of movie similarity scores and sort them in descending order
    cos_sim = create_rating_matrix(movie_data)
    similar_movies = list(enumerate(cos_sim[movie_index]))
    sorted_recc = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    #print(sorted_recc[:5])
    return sorted_recc


#Create the three descriptive methods given the scores and titles

def get_boxplot(scores_list):
    box_recc = sns.boxplot(x=scores_list[1:])
    box_recc.set_title("Quartile spread of the top 5 most relevant results")
    plt.show()
    return None

def get_barchart(scores_list):
    movie_min = min(scores_list)
    movie_max = max(scores_list)
    movie_avg = sum(scores_list) / len(scores_list)
    movie_scores_list = [movie_min, movie_avg, movie_max]
    chart_titles = ["Minimum Score", "Average Score", "Maximum Score"]
    bar_recc = sns.barplot(x=chart_titles, y=movie_scores_list)
    bar_recc.set_title("Min, avg, and max of the top 5 most relevant results")
    plt.show()
    return None

def get_ratings_graph(movie_data):
    #sort titles by most popular
    most_popular = list(movie_data.sort_values(by=['vote_average'], ascending=False)["title"].head(10))
    average_rating = []
    id_title = []
    #retrieve the scores for the top 10 most popular titles
    for f in most_popular:
        temp_score = movie_data[movie_data.title == f]['vote_average'].values[0]
        average_rating.append(temp_score)
        id_title.append(get_index(movie_data, f))
        print(f"ID: {get_index(movie_data, f)} | Title: {f} | Score: {temp_score}")
    sns.set(rc={"figure.figsize": (10, 7)})
    top_ten_graph = sns.barplot(x=id_title, y=average_rating)

    top_ten_graph.set_title("Top 10 films and their ratings. (ID to title printed in console)")
    plt.show()
    return None

def main():
    user_input = ""
    while user_input != "q":
        #try:
        # Create logger string format
        log_format = "%(levelname)s %(asctime)s -- %(message)s"
        # Set up a logger
        logging.basicConfig(filename="log_data.log", level=logging.WARNING, format=log_format)
        logger = logging.getLogger()

        # Read the CSV file
        movie_data = pd.read_csv("movie_dataset.csv")
        # print(movie_data.columns)
        # print(movie_data.head())

        # Create a list of features to be extracted from the dataset
        features = ["genres", "keywords", "cast"]
        vote_count_list = ["vote_average"]

        # Take the list of selected features, clean the data, and append a new row using these combined features
        for f in features:
            movie_data[f] = movie_data[f].fillna('')

        # Clean the vote data
        for v in vote_count_list:
            movie_data[v] = movie_data[v].fillna(0.0)

        #add the new column
        movie_data["added_features"] = movie_data.apply(add_features, axis=1)
        #print(movie_data.head())

        recc_titles = get_recc_titles(movie_data)
        #print(recc_titles[0:5])
        #print(type(recc_titles))

        # Display the top 5 recommended titles
        scores_list = []
        titles_list = []
        i = 0
        for movie in recc_titles:
            title = get_title(movie_data, movie[0])
            score = movie[1]
            titles_list.append(title)
            scores_list.append(score)
            i += 1
            if i > 5:
                break
        #print(titles_list)

        # Retrieve the correct title then pop the same title and score from both lists:
        final_title = titles_list[0]
        titles_list.pop(0)
        scores_list.pop(0)

        # Display the recommended titles
        print(f"Here are some similar movies to {final_title} (From most to least relevant): ")
        j = 0
        while j < 5:
            print(f"{j + 1}. {titles_list[j]}")
            j += 1

        #data_prompt(scores_list, movie_data)

        plt.ion()
        while user_input != "q":
            user_input = input("Enter 1 for a quartile spread of scores, 2 for a bar chart of scores, 3 for a bar chart of top 10 movies overall, 4 to search for another title, or q to exit: ")
            if user_input == "1":
                get_boxplot(scores_list)
                plt.waitforbuttonpress()
            elif user_input == "2":
                get_barchart(scores_list)
                plt.waitforbuttonpress()
            elif user_input == "3":
                get_ratings_graph(movie_data)
                plt.waitforbuttonpress()
            elif user_input == "4":
                break

        #except TypeError:
            #print("Something has gone wrong.")




if __name__ == "__main__":
    main()
