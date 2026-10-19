"""Serve the Module 4 API contract with its verified exported model."""
import uvicorn
from core import load_model, make_api

model, manifest = load_model()
app = make_api(model, manifest)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
