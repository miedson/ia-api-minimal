from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http import models

app = FastAPI()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = qdrant_client.QdrantClient(host="qdrant", port=6333)

COLLECTION = "default"

client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

class Text(BaseModel):
    id: int
    text: str

class Query(BaseModel):
    text: str

@app.post("/embed")
def embed(data: Text):
    vector = model.encode(data.text).tolist()

    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=data.id,
                vector=vector,
                payload={"text": data.text}
            )
        ],
    )
    return {"status": "ok"}

@app.post("/search")
def search(q: Query):
    vector = model.encode(q.text).tolist()

    hits = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=5,
        with_payload=True
    )

    return [
        {
            "id": p.id,
            "score": p.score,
            "text": p.payload["text"]
        }
        for p in hits.points
    ]
