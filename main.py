import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Smart To-Do Backend")

# Enable CORS (HTML file-il ninnoo direct access tharaan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML Model
try:
    model = joblib.load("priority_model.pkl")
except:
    model = None

# In-memory storage
todo_db = []


class TaskRequest(BaseModel):
    title: str


@app.get("/")
def home():
    return {"status": "Smart To-Do API is Running"}


@app.post("/tasks/")
def create_task(task: TaskRequest):
    if model:
        predicted_priority = model.predict([task.title])[0]
    else:
        predicted_priority = "Medium"

    new_task = {
        "id": len(todo_db) + 1,
        "title": task.title,
        "priority": predicted_priority,
    }

    todo_db.append(new_task)
    return {"message": "Task added!", "task": new_task}


@app.get("/tasks/")
def get_tasks():
    return {"tasks": todo_db}