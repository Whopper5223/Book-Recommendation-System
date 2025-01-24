# Book-Recommendation-System
A personalized book recommendation system using TF-IDF, cosine similarity, and collaborative filtering. It processes large datasets to recommend books based on user preferences and historical interactions. Built with Python, Pandas, Scikit-learn, and real Goodreads data, it highlights machine learning and data science expertise.

Personalized Book Recommendation System
This project provides personalized book recommendations tailored to user preferences and past interactions. Leveraging TF-IDF, cosine similarity, and collaborative filtering, it showcases proficiency in data processing and handling large, real-world datasets.

Dataset
The dataset was sourced from Mengting Wan's Goodreads dataset repository (https://mengtingwan.github.io/data/goodreads.html) since the official Goodreads API has been discontinued, making direct data access impossible. The dataset includes book metadata, user interactions, and mappings. Due to the large size of some files, only essential smaller files are included here. For the full dataset, download directly from the repository.

Requirements
To run the system, download the following files from Mengting's repository:

goodreads_books.json.gz: Detailed book graph (~2GB, ~2.3M books).

goodreads_interactions.csv: Complete user-book interactions (~4.1GB).

book_id_map.csv: Book IDs reconstructed by joining this file.

Place these files in the same folder as the provided Python scripts. Once set up, the system will be fully operational.

