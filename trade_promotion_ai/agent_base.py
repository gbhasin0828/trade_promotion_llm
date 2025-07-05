"""
Base classes for all AI agents in the Trade Promotion System

File: trade_promotion_ai/orchestrator/agent_base.py
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message format for agent communication"""
    id: str
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    
    @classmethod
    def create(cls, sender: str, receiver: str, message_type: str, content: Dict[str, Any]):
        """Create a new message"""
        return cls(
            id=str(uuid.uuid4()),
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.now()
        )


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.status = "idle"
        self.logger = logging.getLogger(f"Agent.{name}")
        self.message_queue = asyncio.Queue()
        self.running = False
        
    async def start(self):
        """Start the agent"""
        self.running = True
        self.status = "running"
        self.logger.info(f"Agent {self.name} started")
        
        # Start message processing loop
        asyncio.create_task(self._process_messages())
    
    async def stop(self):
        """Stop the agent"""
        self.running = False
        self.status = "stopped"
        self.logger.info(f"Agent {self.name} stopped")
    
    async def send_message(self, receiver: str, message_type: str, content: Dict[str, Any]) -> str:
        """Send message to another agent"""
        message = AgentMessage.create(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            content=content
        )
        
        self.logger.info(f"Sending {message_type} to {receiver}")
        # In real implementation, this would go through message broker
        return message.id
    
    async def receive_message(self, message: AgentMessage):
        """Receive message from another agent"""
        await self.message_queue.put(message)
    
    async def _process_messages(self):
        """Process incoming messages"""
        while self.running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                self.logger.info(f"Processing {message.message_type} from {message.sender}")
                await self.handle_message(message)
                
            except asyncio.TimeoutError:
                # No message received, continue
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle incoming message - must be implemented by subclass"""
        pass
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process main request - must be implemented by subclass"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "name": self.name,
            "status": self.status,
            "running": self.running,
            "queue_size": self.message_queue.qsize()
        }


class AgentManager:
    """Manages all agents in the system"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("AgentManager")
    
    def register_agent(self, agent: BaseAgent):
        """Register a new agent"""
        self.agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    async def start_all_agents(self):
        """Start all registered agents"""
        for agent in self.agents.values():
            await agent.start()
        self.logger.info("All agents started")
    
    async def stop_all_agents(self):
        """Stop all agents"""
        for agent in self.agents.values():
            await agent.stop()
        self.logger.info("All agents stopped")
    
    async def route_message(self, message: AgentMessage):
        """Route message to target agent"""
        if message.receiver in self.agents:
            await self.agents[message.receiver].receive_message(message)
        else:
            self.logger.error(f"Agent {message.receiver} not found")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "total_agents": len(self.agents),
            "agents": {name: agent.get_status() for name, agent in self.agents.items()}
        }


# Example specialized agent
class QueryAgent(BaseAgent):
    """Example agent that handles user queries"""
    
    async def handle_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle incoming message"""
        if message.message_type == "parse_query":
            return await self._parse_query(message.content.get("query", ""))
        return {"error": "Unknown message type"}
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process user query"""
        query = request.get("query", "")
        self.logger.info(f"Processing query: {query}")
        
        # Simple query processing (will be enhanced later)
        result = {
            "original_query": query,
            "intent": "unknown",
            "entities": [],
            "processed": True
        }
        
        return result
    
    async def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse user query - simple implementation"""
        return {
            "query": query,
            "intent": "analysis" if "show" in query.lower() else "optimization",
            "entities": []
        }


# Testing
if __name__ == "__main__":
    async def test_agents():
        # Create agent manager
        manager = AgentManager()
        
        # Create and register agents
        query_agent = QueryAgent("query_agent")
        manager.register_agent(query_agent)
        
        # Start agents
        await manager.start_all_agents()
        
        # Test query processing
        result = await query_agent.process_request({
            "query": "Show me the best promotions for Product A"
        })
        
        print("Query Result:", result)
        print("System Status:", manager.get_system_status())
        
        # Stop agents
        await manager.stop_all_agents()
    
    # Run test
    asyncio.run(test_agents())