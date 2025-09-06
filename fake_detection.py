import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score

nltk.download('stopwords')
stop=stopwords.words('english')

#Function to clean text
def clean_text(text):
    text=text.lower()
    text=re.sub(r'\W','',text)
    text=''.join([word for word in text.split() if word not in stop])
    return text

#Load dataset
df=pd.read_csv("fake_or_real_news.csv")
df['text']=df['text'].apply(clean_text)

#TF-IDF features
tfidf=TfidfVectorizer(max_features=5000)
X=tfidf.fit_transform(df['text'])
y=df['label']

#Train/test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Train model
model=LogisticRegression()
model.fit(X_train,y_train)

#Evaluate
y_pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#Predict function
def predict_fake(text):
    text=clean_text(text)
    vector=tfidf.transform([text])
    return model.predict(vector)[0]

#Example usage
sample text="Breaking news: AI can now detect fake news!"
print(predict_fake(sample_text))