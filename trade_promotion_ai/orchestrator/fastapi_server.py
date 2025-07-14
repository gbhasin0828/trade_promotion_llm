"""
Minimal FastAPI server for Trade Promotion AI system

File: trade_promotion_ai/orchestrator/fastapi_server.py
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import logging

from agent_base import AgentManager, QueryAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Trade Promotion AI")

# Global agent manager
agent_manager = AgentManager()


# Request/Response models
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    success: bool
    result: Dict[str, Any]


# Initialize agents on startup
@app.on_event("startup")
async def startup():
    """Start agents when server starts"""
    query_agent = QueryAgent("query_agent")
    agent_manager.register_agent(query_agent)
    await agent_manager.start_all_agents()
    logger.info("System ready!")


# Health check
@app.get("/health")
async def health():
    """Check if system is working"""
    return {"status": "ok"}


# Main endpoint - process user queries
@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Handle user queries"""
    
    # Get the query agent
    query_agent = agent_manager.agents["query_agent"]
    
    # Process the query
    result = await query_agent.process_request({
        "query": request.query
    })
    
    return QueryResponse(
        success=True,
        result=result
    )


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=True)