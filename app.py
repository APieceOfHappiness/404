import uvicorn
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd

classes = pd.read_csv('./data.csv')

X_train = classes['name'] 
y_train = classes['book'] 
 
russian_stopwords = ['см','толщина','мм','диаметр','длина','м','из','кг','м3', 'объем','класс','h','мм2','до','от','вт','квт','лс','т','HH','ч','c','мин'] 
 
tfidf = TfidfVectorizer(lowercase=True, stop_words=russian_stopwords) 
X_train_tfidf = tfidf.fit_transform(X_train) 
 
knn = KNeighborsClassifier(n_neighbors=2) 
knn.fit(X_train_tfidf, y_train) 
 
in_books = {} 
 
for book in classes['book'].unique(): 
    cur_df = classes[classes['book'] == book] 
 
    in_books[book] = (KNeighborsClassifier(n_neighbors=2), TfidfVectorizer(lowercase=True, stop_words=russian_stopwords)) 
     
    X_train_cur = in_books[book][1].fit_transform(cur_df['name']) 
    in_books[book][0].fit(X_train_cur, cur_df['part']) 
    print(f"Book {book} complete") 
 
in_parts = {} 
 
for book in classes['book'].unique(): 
    cur1_df = classes[classes['book'] == book] 
    in_parts[book] = {} 
    for part in cur1_df['part'].unique(): 
        cur_df = cur1_df[cur1_df['part'] == part] 
 
        in_parts[book][part] = (KNeighborsClassifier(n_neighbors=3), TfidfVectorizer(lowercase=True, stop_words=russian_stopwords)) 
        X_train_cur = in_parts[book][part][1].fit_transform(cur_df['name']) 
        in_parts[book][part][0].fit(X_train_cur, cur_df['resource_code'])

router = APIRouter()

@router.post("/submissions")
def get_images(test: list[str]):    
    test_tfidf = tfidf.transform(test) 
    book_pred = knn.predict(test_tfidf) 
    
    part_pred = [] 
    
    for i, j in zip(book_pred, test): 
        current_sample = in_books[i][1].transform([j]) 
    
        part_pred.append(in_books[i][0].predict(current_sample)[0]) 
    
    targ_pred = [] 
    
    for i, j, k in zip(book_pred, test, part_pred): 
        current_sample = in_parts[i][k][1].transform([j]) 
    
        targ_pred.append(in_parts[i][k][0].predict(current_sample)[0])
    return targ_pred

app = FastAPI(
        docs_url="/api/docs",
        title="HACK"
    )
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)