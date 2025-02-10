from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import uvicorn

# Load the fine-tuned model
MODEL_PATH = "./fine_tuned_t5"  # Local path in your project folder
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Initialize FastAPI
app = FastAPI()

# Define request format
class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150

# Summarization function
def generate_summary(text, max_length=150):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Define API route
@app.post("/summarize/")
async def summarize(request: SummaryRequest):
    summary = generate_summary(request.text, request.max_length)
    return {"summary": summary}

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
