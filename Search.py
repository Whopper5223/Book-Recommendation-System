import pandas as pd
import gzip
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from scipy.sparse import coo_matrix

# Step 1: Load the cleaned data from books.json
titles = pd.read_json("books.json")
# print("Loaded titles data:", titles.head())  # Debugging: Check the titles data

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(titles["propertitle"])

# Enable showing all columns in terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Function to format each book with a clickable hyperlink
def format_book(book):
    title = book['title']
    ratings = book['ratingnum']
    bookid = book['bookid']
    url = book['url']
    clickable_url = f"\033]8;;{url}\033\\Goodreads\033]8;;\033\\"
    return f"Title: {title}\nRatings: {ratings}\nLink: {clickable_url}\nBookID: {bookid}\n"

def format_recs(book):
    title = book['title']
    ratings = book['ratingnum']
    bookid = book['bookid']
    url = book['url']
    score = book['score']
    clickable_url = f"\033]8;;{url}\033\\Goodreads\033]8;;\033\\"
    return f"Title: {title}\nRatings: {ratings}\nLink: {clickable_url}\nBookID: {bookid}\nScore: {score}\n"

# Search function
def search(booktitle, vectorizer):
    p_booktitle = re.sub("[^a-zA-Z0-9 ]", "", booktitle.lower())
    v_booktitle = vectorizer.transform([p_booktitle])
    cos_sim = cosine_similarity(v_booktitle, tfidf).flatten()
    bookindex = np.argpartition(cos_sim, -10)[-10:]
    sim_titles = titles.iloc[bookindex]
    sim_titles = sim_titles.sort_values("ratingnum", ascending=False)
    # print("Search results:", sim_titles)  # Debugging: Check search results
    return sim_titles.head(10)

def save_liked_books():
    liked_books = []  # List to store liked books

    while True:
        query = input("Would you like to add a book? (yes/no): ").strip()
        if query.lower() == 'no':
            break

        elif query.lower() == 'yes':
            book_title = input("Enter the book title: ").strip()
            results = search(book_title, vectorizer)
            
            print("\nSearch Results:")
            for index, (_, book) in enumerate(results.iterrows(), start=1):
                print(f"{index}. {format_book(book)}")

            indices = input("Enter the numbers of books you like (comma-separated): ").strip()
            if indices:
                try:
                    selected_indices = [int(i) - 1 for i in indices.split(",")]
                    selected_books = results.iloc[selected_indices].to_dict(orient="records")
                    ratings_input = input("Enter your ratings for these books (comma-separated, out of 5): ").strip()
                    ratings = [float(r) for r in ratings_input.split(",")]

                    if len(ratings) != len(selected_books):
                        print("Error: The number of ratings does not match the number of selected books.")
                        continue

                    for book, rating in zip(selected_books, ratings):
                        book["rating"] = rating
                        book["user_id"] = -1  
                    liked_books.extend(selected_books)

                except Exception as e:
                    print(f"Error processing selection or ratings: {e}")
        else:
            print("Please enter 'yes' or 'no'.")

    if liked_books:
        liked_books_df = pd.DataFrame(liked_books)
        # print("Liked books DataFrame before saving:", liked_books_df.head())  # Debugging: Check liked books
        liked_books_df.to_csv("liked_books.csv", index=False)
        print("\nLiked books saved to liked_books.csv!")
    else:
        print("\nNo books were selected.")

liked = ["9317691", "8153988", "20494944"]

def recommend(liked_books):
    bookmap = {}
    with open("book_id_map.csv", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            int_id, bookid = line.strip().split(",")
            bookmap[int_id] = bookid
    # print(f"Bookmap size: {len(bookmap)}")  # Debugging: Check bookmap size

    overlap = {}
    with open("goodreads_interactions.csv", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            user_id, int_id, _, rating, _ = line.strip().split(",")
            bookid = bookmap.get(int_id)
            if bookid in liked:
                overlap[user_id] = overlap.get(user_id, 0) + 1
    # print(f"Overlap size: {len(overlap)}")  # Debugging: Check overlap size

    overlap = {user_id: count for user_id, count in overlap.items() if count > liked_books.shape[0] / 15}
    print("Initial overlap:", overlap)

    overlap_set = set(overlap)
    # print(f"Filtered overlap set size: {len(overlap_set)}")  # Debugging: Check filtered overlap set

    bookrecs = []
    with open("goodreads_interactions.csv", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            user_id, int_id, _, rating, _ = line.strip().split(",")
            if user_id in overlap_set:
                bookid = bookmap.get(int_id)
                if bookid:
                    bookrecs.append([user_id, bookid, rating])
    # print(f"Book recommendations size: {len(bookrecs)}")  # Debugging: Check book recommendations

    recoms = pd.DataFrame(bookrecs, columns=["user_id", "bookid", "rating"])
    recoms = pd.concat([liked_books[["user_id", "bookid", "rating"]], recoms])
    # print("Recoms DataFrame:", recoms.head())  # Debugging: Check recoms DataFrame

    recoms["bookid"] = recoms["bookid"].astype(str)
    recoms["user_id"] = recoms["user_id"].astype(str)
    recoms["rating"] = pd.to_numeric(recoms["rating"])

    recoms["u_index"] = recoms["user_id"].astype("category").cat.codes
    recoms["b_index"] = recoms["bookid"].astype("category").cat.codes
    recom_coo = coo_matrix((recoms["rating"], (recoms["u_index"], recoms["b_index"])))
    recom_csr = recom_coo.tocsr()
    myindex = 0
    sim = cosine_similarity(recom_csr[myindex, :], recom_csr).flatten()
    # print(f"Similarity scores size: {len(sim)}")  # Debugging: Check similarity scores

    top_n = min(15, len(sim))
    sim_users_index = np.argpartition(sim, -top_n)[-top_n:]
    # print(f"Top similar users index: {sim_users_index}")  # Debugging: Check similar users

    sim_users = recoms[recoms["u_index"].isin(sim_users_index)].copy()
    sim_users = sim_users[sim_users["user_id"] != "-1"]

    titles2 = pd.read_json("books.json")
    titles2["bookid"] = titles2["bookid"].astype(str)
    bookrecs = sim_users.groupby("bookid").rating.agg(["count", "mean"])
    bookrecs = bookrecs.merge(titles2, how="inner", on="bookid")
    bookrecs["adj_count"] = bookrecs["count"] * (bookrecs["count"] / bookrecs["ratingnum"])
    bookrecs["score"] = bookrecs["mean"] * bookrecs["adj_count"]
    bookrecs = bookrecs[bookrecs["count"] > 2]
    bookrecs = bookrecs[bookrecs["mean"] > 4]
    toprecs = bookrecs.sort_values("score", ascending=False)
    # print("Top recommendations DataFrame:", toprecs.head())  # Debugging: Check final recommendations

    print("\nTop Recommendations:")
    for _, book in toprecs.head(10).iterrows():
        print(format_book(book))

if __name__ == "__main__":
    save_liked_books()
    liked_books = pd.read_csv("liked_books.csv")
    # print("Liked books loaded:", liked_books.head())  # Debugging: Check liked_books DataFrame
    recommend(liked_books)
