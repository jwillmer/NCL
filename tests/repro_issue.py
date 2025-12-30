
import json
from fastapi import FastAPI
from ncl.api.conversations import router

app = FastAPI()
app.include_router(router)

schema = app.openapi()
paths = schema.get("paths", {})
post_conv = paths.get("/conversations", {}).get("post", {})
parameters = post_conv.get("parameters", [])

print("Parameters for POST /conversations:")
found = False
for param in parameters:
    print(f"- {param['name']} ({param['in']})")
    if param['name'] == 'request' and param['in'] == 'query':
        found = True

if found:
    print("\nISSUE REPRODUCED: 'request' is a query parameter!")
else:
    print("\nIssue NOT reproduced in minimal setup.")
