from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from src.routes import predict
app = FastAPI()
app.include_router(predict.router)
origins = [
    "http://localhost:8000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Start Page"}
