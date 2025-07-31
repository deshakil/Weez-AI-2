from fastapi import FastAPI
from routes.ai_routes import router as ai_router

app = FastAPI(title="Weez MCP Agent", version="1.0.0")

# Include the AI router
app.include_router(ai_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
