from django.shortcuts import render

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def home(request):
    return render(request, 'recommend.html')

def result(request):
    if request.method == 'POST':
        random_number = int(request.POST['SongNumber'])
        songs = pd.read_csv('./music_recommender/songdata.csv')
        songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
        songs['text'] = songs['text'].str.replace(r'\n', '')
        tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
        lyrics_matrix = tfidf.fit_transform(songs['text'])
        cosine_similarities = cosine_similarity(lyrics_matrix)
        similarities = {}
        for i in range(len(cosine_similarities)):
            # Now we'll sort each element in cosine_similarities and get the indexes of the songs.
            similar_indices = cosine_similarities[i].argsort()[:-50:-1]
            # After that, we'll store in similarities each name of the 50 most similar songs.
            # Except the first one that is the same song.
            similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]
        class ContentBasedRecommender:
            def __init__(self, matrix):
                self.matrix_similar = matrix
                
            def recommend(self, recommendation):
                # Get song to find recommendations for
                song = recommendation['song']
                # Get number of songs to recommend
                number_songs = recommendation['number_songs']
                # Get the number of songs most similars from matrix similarities
                recom_song = self.matrix_similar[song][:number_songs]
                # return each item
                result_dict = {}
                for i in recom_song:
                    result_dict[str(i[1])] = {"score":i[0], "name":i[1], "artist":i[2]}
                return result_dict

        recommedations = ContentBasedRecommender(similarities)
        recommendation = {
            "song": songs['song'].iloc[random_number],
            "number_songs": 4 
        }
        recommended = recommedations.recommend(recommendation)

        return render(request, 'result.html', {'result':recommended.values, 'recommendation':recommendation})
    return render(request, 'recommend.html')