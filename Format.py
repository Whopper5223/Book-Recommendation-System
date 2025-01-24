#imports 
import gzip
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


#Step 1: creating a way to search for a book.
#use gzip so that we dont use too much memory as this opens it line by line instead of the whole thing
with gzip.open("goodreads_books.json.gz", 'r') as f:
    line = f.readline()

print(json.loads(line))


def get_data(line):
    data= json.loads(line)
    return{
        "bookid": data["book_id"],
        "title": data["title_without_series"],
        "url": data["url"],
        "ratingnum": data["ratings_count"],
        "cover": data["image_url"]
    }
    
    
books=[]
with gzip.open("goodreads_books.json.gz", 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        data = get_data(line)
        
        try: 
            ratings = int(data["ratingnum"])
        except:
            continue
        if ratings >30:
            books.append(data)

titles = pd.DataFrame.from_dict(books)
titles["ratingnum"]= pd.to_numeric(titles["ratingnum"])
titles["propertitle"] = titles["title"].str.replace("[^a-zA-Z0-9 ]", "", regex = True)
titles["propertitle"] = titles["propertitle"].str.lower()
titles["propertitle"] = titles["propertitle"].str.replace("\s+", " ", regex=True)
print(titles)

titles = titles[titles["propertitle"].str.len()>0]
titles.to_json("books.json")


