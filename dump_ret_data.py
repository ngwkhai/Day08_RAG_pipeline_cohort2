import os
from dotenv import load_dotenv
import time
import json
from pathlib import Path

load_dotenv()
pageindex_key = os.getenv("PAGEINDEX_API_KEY")

from pageindex import PageIndexClient
pi = PageIndexClient(api_key=pageindex_key)

docs = pi.list_documents(limit=50).get('documents', [])
print(f"Found {len(docs)} documents.")

for doc in docs:
    if doc.get('status') == 'completed':
        doc_id = doc['id']
        print(f"Querying doc {doc_id}...")
        q_res = pi.submit_query(doc_id=doc_id, query="hình phạt", thinking=False)
        ret_id = q_res.get('retrieval_id')
        print(f"Retrieval ID: {ret_id}")
        
        for _ in range(15):
            time.sleep(2)
            ret_data = pi.get_retrieval(ret_id)
            status = ret_data.get('status')
            print(f"Ret Status: {status}")
            if status == 'completed':
                with open('ret_data.json', 'w', encoding='utf-8') as f:
                    json.dump(ret_data, f, ensure_ascii=False, indent=2)
                print("Saved to ret_data.json")
                exit(0)
