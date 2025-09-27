import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import json

# -------------------------
# Actor / CSV Tool
# -------------------------
class Actor:
    def __init__(self, csv_file="data/prices.csv"):
        self.df = pd.read_csv(csv_file, skipinitialspace=True)

    def lookup_price(self, product):
        row = self.df[self.df['product_name'].str.lower() == product.lower()]
        if not row.empty:
            return float(row['price'].values[0])
        return "Price not found"

# -------------------------
# Reasoner (Dummy / Rule-based)
# -------------------------
class Reasoner:
    def decide(self, query, retrieval_results):
        if "price" in query.lower():
            return "USE_TOOL"
        return "ANSWER_FROM_KB"

# -------------------------
# Retriever
# -------------------------
class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)

    def search(self, query, top_k=3):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        results = [{"doc": self.docs[i], "score": float(scores[i])} for i in top_indices]
        return results

# -------------------------
# Controller
# -------------------------
class Controller:
    def __init__(self, retriever, reasoner, actor):
        self.retriever = retriever
        self.reasoner = reasoner
        self.actor = actor

    def handle_query(self, query):
        start = time.time()
        retrieval = self.retriever.search(query)
        retrieval_latency = round((time.time()-start)*1000, 2)

        start = time.time()
        decision = self.reasoner.decide(query, retrieval)
        decision_latency = round((time.time()-start)*1000, 2)

        tool_result = None
        if decision == "USE_TOOL":
            start = time.time()
            # Extract product name from query (simplified)
            product = query.replace("What is the price of ", "").replace("?", "").strip()
            tool_result = self.actor.lookup_price(product)
            tool_latency = round((time.time()-start)*1000, 2)
            final_answer = f"The price of {product} is {tool_result}"
        else:
            tool_latency = None
            final_answer = f"Answer (from KB): {retrieval[0]['doc']}"

        log = {
            "query": query,
            "retrieval": retrieval,
            "retrieval_latency_ms": retrieval_latency,
            "decision": decision,
            "decision_latency_ms": decision_latency,
            "tool_call": {"tool": "PriceLookup", "item": product, "result": tool_result} if tool_result else None,
            "tool_latency_ms": tool_latency,
            "final_answer": final_answer
        }

        print(json.dumps(log, indent=2))
        print("[Final Answer]", final_answer)
        return final_answer

# -------------------------
# Demo
# -------------------------
def run_demo():
    docs = [
        "The Acme Lamp is a stylish desk lamp for modern offices.",
        "The Acme Chair comes in different colors.",
        "Customer reviews show Acme Lamp is very popular.",
        "Acme products are known for their quality and affordability.",
        "The Acme Chair is designed for comfort and durability.",
        "Acme Desk provides plenty of workspace for students.",
        "Accessories are often bundled with Acme furniture."
    ]

    retriever = Retriever(docs)
    reasoner = Reasoner()
    actor = Actor(csv_file="data/prices.csv")
    ctr = Controller(retriever, reasoner, actor)

    test_queries = [
        "What is the price of Acme Lamp?",
        "Tell me about the Acme Chair.",
        "Who makes Acme products?"
    ]

    for q in test_queries:
        ctr.handle_query(q)

if __name__ == "__main__":
    run_demo()
