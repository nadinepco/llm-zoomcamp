# 3. Vector Search
## Table of Contents
- [Semantic Search](#31-semantic-search)
- [Evaluating Retrieval](#33-evaluating-retrieval)
    - [Getting ground truth data](#332-getting-ground-truth-data)
    - [Evaluation Metrics](#323-evaluation-metrics)
    - [Ranking evaluation: text search](#324-ranking-evaluation-text-search)
    - [Ranking evaluation: Vector search](#325-ranking-evaluation-vector-search)

## 3.1 Semantic Search with Elastic search
* Two very important concepts in Elasticsearch are documents and indexes.

* A document is collection of fields with their associated values. 

* To work with Elasticsearch you have to organize your data into documents, and then add all your documents to an index. 

* Index as a collection of documents that is stored in a highly optimized format designed to perform efficient searches.

### Sentence Transformers
Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art text and image embedding models.
Through sentence transformers package, you can call the pre-trained models in just one line of code.

```
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2")
```

### Run Elastic search on Docker
```
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \ 
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

### Basic Semantic Search
- Scores are between 0 and 1
```
search_term = "windows or mac?"
vector_search_term = model.encode(search_term)

query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000, 
}

res = es_client.search(
    index=index_name, 
    knn=query, 
    source=["text", "section", "question", "course"])
res["hits"]["hits"]
```

### Advanced Semantic Search
- Scores have values greater than 1
```
# Included "knn" in the search query (to perform a semantic search) along with the filter  
knn_query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000
}

response = es_client.search(
    index=index_name,
    query={
        "match": {"section": "General course-related questions"},
    },
    knn=knn_query,
    size=5
)
```

## 3.2 Evaluating Retrieval
### 3.2.1 Introduction
* To be able to evaluate the retrieval, typically we need a gold standard data set or "Ground Truth" dataset for each query. \
    ```
    Example:
    Query: I just discovered the course. Can I still join?
    Relevant documents: doc1 doc4 doc 11
    ```

### 3.2.2 Getting ground truth data
* Some ways to generating ground truth data:
    * Humans annotators
    * Observing what are the usual questions asked by users and sees what the system returns and evaluate the result
    * LLMs - generate questions for each record

* Assigning unique ID for each document
    * Can't simply assign numbers to reach document because it will change everytime the FAQs are updated
    * One way is to use IDs of each section (google word document)
    * Another way to generate document id: combination of course, question and text, then create a hashobject for it and get the first 8 characters
        ```python
        import hashlib

        def generate_document_id(doc):
            combined = f"{doc['course']}-{doc['question']}-{doc['text'][:10]}"
            hash_object = hashlib.md5(combined.encode())
            hash_hex = hash_object.hexdigest()
            document_id = hash_hex[:8]
            return document_id
        ```
* Generate questions
    ```python
    prompt_template = """
    You emulate a student who's taking our course.
    Formulate 5 questions this student might ask based on a FAQ record. The record
    should contain the answer to the questions, and the questions should be complete and not too short.
    If possible, use as fewer words as possible from the record. 

    The record:

    section: {section}
    question: {question}
    answer: {text}

    Provide the output in parsable JSON without using code blocks:

    ["question1", "question2", ..., "question5"]
    """.strip()
    ```

    ```python
    from openai import OpenAI
    client = OpenAI()
    def generate_questions(doc):
        prompt = prompt_template.format(**doc)

        response = client.chat.completions.create(
            model='gpt-4o',
            messages=[{"role": "user", "content": prompt}]
        )

        json_response = response.choices[0].message.content
        return json_response
    ```
    [Getting ground truth data set notebook](https://github.com/nadinepco/llm-zoomcamp/blob/ad3fcab7667e1738431edc13c3dec54c46356f47/03-vector-search/eval/ground-truth-data.ipynb)


### 3.2.3 Evaluation Metrics
* Metrics to be used: 
    * Hit rate (recall) - if the relevant document is retrieved or not
        ```
        Example:
        Relevance                              score
        [true, false, false, false, false]  -> 1
        [false, false, false, true, false]  -> 1
        [false, false, false, false, false] -> 0
        [false, true, false, false, false]  -> 1

        hit rate = relevance total / # of queries
        hit rate = 3 / 4
        hit rate = 0.75
        ```
        > [!NOTE]
        > In hit rate, the position of the True value is not important. As long as the correct document is included in the top n rank. 

        ```python
        def hit_rate(relevance_total):
            cnt = 0

            for line in relevance_total:
                if True in line:
                    cnt = cnt + 1

            return cnt / len(relevance_total)
        ```

    * Mean Reciprocal Rank (mrr) - how good is the ranking (relevant documents should be at the top, higher metric value means better ranking)
        ```
        Example:
        Relevance                              score
        [true, false, false, false, false]  -> 1
        [false, false, false, true, false]  -> 1/4 = 0.25
        [false, false, false, false, false] -> 0
        [false, true, false, false, false]  -> 1/2 = 0.5

        score = 1 / position
        mrr = total_score / # of queries
        mrr = (1+0.25+0.5) / 4
        mrr = 0.4375
        ```
        > [!NOTE]
        > In mrr, the position of the True value (first relevant document) is considered.
        ```python
        def mrr(relevance_total):
            total_score = 0.0

            for line in relevance_total:
                for rank in range(len(line)):
                    if line[rank] == True:
                        total_score = total_score + 1 / (rank + 1)

            return total_score / len(relevance_total)
        ```
### 3.2.4 [Ranking evaluation: text search](https://github.com/nadinepco/llm-zoomcamp/blob/ad3fcab7667e1738431edc13c3dec54c46356f47/03-vector-search/eval/evaluate-text.ipynb)
#### Elastic search mapping / index settings
```python
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') # make sure elasticsearch is running

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
        }
    }
}

index_name = "course-questions"

# create index
es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

# apply index to each document
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
```
#### Elastic search
```python
def elastic_search(query, course):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs

elastic_search(
    query="I just discovered the course. Can I still join?",
    course="data-engineering-zoomcamp"
)
```
<details>
<summary>Result</summary>

```
[{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp',
  'id': '7842b56a'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp',
  'id': '63394d91'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp',
  'id': 'a482086d'},
 {'text': 'Yes, the slack channel remains open and you can ask questions there. But always sDocker containers exit code w search the channel first and second, check the FAQ (this document), most likely all your questions are already answered here.\nYou can also tag the bot @ZoomcampQABot to help you conduct the search, but don’t rely on its answers 100%, it is pretty good though.',
  'section': 'General course-related questions',
  'question': 'Course - Can I get support if I take the course in the self-paced mode?',
  'course': 'data-engineering-zoomcamp',
  'id': 'eb56ae98'},
 {'text': "You don't need it. You're accepted. You can also just start learning and submitting homework without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.",
  'section': 'General course-related questions',
  'question': 'Course - I have registered for the Data Engineering Bootcamp. When can I expect to receive the confirmation email?',
  'course': 'data-engineering-zoomcamp',
  'id': '0bbf41ec'}]
```
</details>

#### Get the relevance total
```python
relevance_total = []

# iterate over the queries and invoke the search function
for q in tqdm(ground_truth):
    doc_id = q['document']
    results = elastic_search(query=q['question'], course=q['course'])
    # check if the id in the top 5 results matches the actual document id for the question
    relevance = [d['id'] == doc_id for d in results]
    relevance_total.append(relevance)

hit_rate(relevance_total)
mrr(relevance_total)
```

### 3.2.5 [Ranking evaluation: Vector search](https://github.com/nadinepco/llm-zoomcamp/blob/ad3fcab7667e1738431edc13c3dec54c46356f47/03-vector-search/eval/evaluate-vector.ipynb)

#### Elastic search mapping / index settings with vector
```python
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') 

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

# Index each document with its embedding
for i, doc in enumerate(machine_learning_documents):
    doc['question_text_vector'] = X[i]
    es_client.index(index=index_name, body=doc)
```
#### Elastic search
```python
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs
```

```python
def question_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = embedding_model.encode(question)

    return elastic_search_knn('question_text_vector', v_q, course)
```

#### Evaluate
```python
def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
    }
```