import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_pages_from_category(category, max_pages=10):
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:{category}&cmlimit={max_pages}&format=json"
    response = requests.get(url).json()
    pages = response['query']['categorymembers']
    return [page['title'] for page in pages]


def get_page_content(title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={title}&format=json"
    response = requests.get(url).json()
    pages = response['query']['pages']
    content = next(iter(pages.values())).get('extract', '')
    return content


nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return tokens


def create_inverted_index(documents):
    inverted_index = defaultdict(list)
    for doc_id, content in enumerate(documents):
        for term in set(content):
            inverted_index[term].append(doc_id)
    return inverted_index


def process_boolean_query(query, index):
    query_terms = re.findall(r'\w+', query.lower())
    operator = 'AND'
    if 'OR' in query.upper():
        operator = 'OR'
    elif 'NOT' in query.upper():
        operator = 'NOT'

    results = set(index[query_terms[0]]) if query_terms[0] in index else set()

    for term in query_terms[1:]:
        if term in ['AND', 'OR', 'NOT']:
            operator = term
        elif operator == 'AND' and term in index:
            results.intersection_update(index[term])
        elif operator == 'OR' and term in index:
            results.update(index[term])
        elif operator == 'NOT' and term in index:
            results.difference_update(index[term])

    return results


def rank_by_tfidf(query_terms, documents):
    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([' '.join(query_terms)])
    cosine_similarities = cosine_similarity(query_vector, doc_matrix)

    ranked_docs = sorted(enumerate(cosine_similarities[0]), key=lambda x: x[1], reverse=True)
    return [(i, score) for i, score in ranked_docs if score > 0]


def search_engine(query, index, documents, titles):
   
    matching_titles = [i for i, title in enumerate(titles) if query.lower() in title.lower()]
    if matching_titles:
        for doc_id in matching_titles:
            print(f"\nTitle: {titles[doc_id]}\n{'-'*len(titles[doc_id])}\n{documents[doc_id]}\n")
        return matching_titles

    
    boolean_results = process_boolean_query(query, index)
    if not boolean_results:
        print("No matching documents found.")
        return []

    results = [(doc_id, 1.0) for doc_id in boolean_results]

    
    results = rank_by_tfidf(query.split(), [documents[i] for i in boolean_results])
    
    for doc_id, score in results:
        accuracy = score * 100
        print(f"\nTitle: {titles[doc_id]}\n{'-'*len(titles[doc_id])}\n{documents[doc_id]}\n\n(Accuracy: {accuracy:.2f}%)\n")
    return results


category = input("Enter Wikipedia category to scrape: ")


pages = []
for term in category.split():
    pages.extend(get_pages_from_category(term, max_pages=20))
pages = list(set(pages))  # Remove duplicates

documents = [get_page_content(page) for page in pages]


vectorizer = TfidfVectorizer()
doc_matrix = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([category])
cosine_similarities = cosine_similarity(query_vector, doc_matrix)


print(f"Found {len(pages)} pages in category '{category}':")
for i, page in enumerate(pages):
    print(f"{page} (Accuracy: {cosine_similarities[0][i] * 100:.2f}%)")


df = pd.DataFrame({'Title': pages, 'Content': documents})
df['Cleaned_Content'] = df['Content'].apply(preprocess_text)
inverted_index = create_inverted_index(df['Cleaned_Content'])


df.to_csv('wikipedia_data.csv', index=False)
print("Wikipedia data saved to wikipedia_data.csv")


query = input("Please select the content you want to read: ")


results = search_engine(query, inverted_index, df['Content'].tolist(), df['Title'].tolist())



if results:
    search_results_df = pd.DataFrame([
        {
            'Title': df.iloc[i]['Title'],
            'Content': df.iloc[i]['Content']
        }
        for r in results
        for i in ([r] if isinstance(r, int) else [r[0]]) 
    ])
    search_results_df.to_csv('search_results.csv', index=False)
    print("Search results saved to search_results.csv")


