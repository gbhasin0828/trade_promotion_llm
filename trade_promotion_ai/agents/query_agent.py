"""
Enhanced Query Agent with Trade Promotion Business Logic

File: trade_promotion_ai/agents/query_agent.py
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from orchestrator.agent_base import BaseAgent, AgentMessage



logger = logging.getLogger(__name__)


class TradePromotionQueryAgent(BaseAgent):
    """Smart agent that understands trade promotion queries and proposes solution steps"""
    
    def __init__(self, name: str):
        super().__init__(name)
        
        # Load business terminology from your files
        self.business_terms = self._load_business_terms()
        
        # Query interpretation templates
        self.solution_templates = {
            'optimization': {
                'description': 'Find optimal promotion strategy',
                'required_data': ['budget', 'products', 'retailers', 'constraints'],
                'steps': [
                    'Load trained ML model and business rules',
                    'Generate candidate promotion scenarios',
                    'Predict demand using XGBoost model',
                    'Optimize allocation within constraints',
                    'Validate business feasibility'
                ]
            },
            'analysis': {
                'description': 'Analyze historical promotion performance',
                'required_data': ['products', 'retailers', 'time_period'],
                'steps': [
                    'Load historical promotion database',
                    'Filter data by specified criteria',
                    'Calculate key metrics (Lift_%, ROI, Inc_Profit)',
                    'Identify top performing promotions',
                    'Generate insights and recommendations'
                ]
            },
            'prediction': {
                'description': 'Predict promotion outcomes',
                'required_data': ['promotion_details', 'products', 'retailers'],
                'steps': [
                    'Load trained ML model',
                    'Process promotion scenario into model features',
                    'Predict Units and Base_Units using XGBoost',
                    'Calculate business metrics and ROI',
                    'Assess prediction confidence and risk'
                ]
            },
            'simulation': {
                'description': 'What-if scenario analysis',
                'required_data': ['current_state', 'proposed_changes'],
                'steps': [
                    'Load current baseline scenario',
                    'Apply proposed changes to create new scenario',
                    'Predict outcomes for both scenarios using ML',
                    'Compare key metrics and calculate impact',
                    'Generate recommendations based on analysis'
                ]
            }
        }
    
    def _load_business_terms(self) -> Dict[str, List[str]]:
        """Load business terminology from your formula files"""
        # This would load from your actual files - simplified for now
        return {
            'financial_metrics': ['roi', 'profit', 'margin', 'trade rate', 'lift', 'incremental profit'],
            'promotion_types': ['discount', 'bogo', '2for$', 'percentage off', 'buy one get one'],
            'retailers': ['walmart', 'target', 'kroger', 'safeway', 'costco', 'retailer', 'customer', 'account'],
            'products': ['product', 'item', 'sku', 'brand'],
            'constraints': ['budget', 'discount limit', 'promotion weeks', 'minimum margin'],
            'time_periods': ['week', 'month', 'quarter', 'year', 'weeks', 'q1', 'q2', 'q3', 'q4']
        }
    
    async def handle_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle messages from other agents"""
        if message.message_type == "user_confirmation":
            return await self._process_user_feedback(message.content)
        return {"status": "received"}
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process user query with full business understanding"""
        query = request.get("query", "")
        self.logger.info(f"Processing trade promotion query: {query}")
        
        # Step 1: Extract business entities
        entities = self._extract_business_entities(query)
        
        # Step 2: Determine primary intent and sub-intents
        intent_analysis = self._analyze_intent(query, entities)
        
        # Step 3: Identify required data and missing information
        data_requirements = self._identify_data_requirements(intent_analysis, entities)
        
        # Step 4: Propose solution steps
        solution_plan = self._create_solution_plan(intent_analysis, entities, data_requirements)
        
        # Step 5: Generate confirmation request for user
        confirmation_request = self._generate_confirmation_request(query, solution_plan)
        
        result = {
            "original_query": query,
            "interpretation": {
                "primary_intent": intent_analysis['primary_intent'],
                "business_objective": intent_analysis['objective'],
                "entities_found": entities,
                "confidence": intent_analysis['confidence']
            },
            "solution_plan": solution_plan,
            "confirmation_request": confirmation_request,
            "status": "awaiting_confirmation"
        }
        
        return result
    
    def _extract_business_entities(self, query: str) -> Dict[str, Any]:
        """Extract trade promotion specific entities"""
        entities = {}
        query_lower = query.lower()
        
        # Financial entities
        budget_match = re.search(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?(?:mm|m|k|million|thousand)?)', query_lower)
        if budget_match:
            entities['budget'] = budget_match.group(1)
        
        # Percentage entities (discounts, margins, etc.)
        percentages = re.findall(r'(\d+(?:\.\d+)?)%', query_lower)
        if percentages:
            entities['percentages'] = percentages
        
        # Time periods
        time_matches = re.findall(r'(\d+)\s+(week|month|quarter)s?', query_lower)
        if time_matches:
            entities['time_periods'] = time_matches
        
        # Products (enhanced to catch various formats)
        product_patterns = [
            r'product[s]?\s+([a-z0-9,\s&]+)',
            r'item[s]?\s+([a-z0-9,\s&]+)',
            r'sku[s]?\s+([a-z0-9,\s&-]+)'
        ]
        products = []
        for pattern in product_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                # Split on common separators
                product_list = re.split(r'[,&\s]+', match.strip())
                products.extend([p.strip() for p in product_list if p.strip()])
        
        if products:
            entities['products'] = list(set(products))  # Remove duplicates
        
        # Retailers/Customers
        retailer_patterns = [
            r'(?:retailer|customer|account|at)\s+([a-z0-9\s]+)',
            r'(walmart|target|kroger|safeway|costco)'
        ]
        retailers = []
        for pattern in retailer_patterns:
            matches = re.findall(pattern, query_lower)
            retailers.extend(matches)
        
        if retailers:
            entities['retailers'] = list(set(retailers))
        
        # Constraints
        constraints = {}
        
        # Max discount constraint
        max_discount = re.search(r'(?:max|maximum|up to)\s+(\d+(?:\.\d+)?)%\s+discount', query_lower)
        if max_discount:
            constraints['max_discount'] = float(max_discount.group(1))
        
        # Max promotion weeks
        max_promo = re.search(r'(?:promote|promotion).+?(?:max|maximum)\s+(\d+)\s+week', query_lower)
        if max_promo:
            constraints['max_promotion_weeks'] = int(max_promo.group(1))
        
        # Trade rate constraints
        trade_rate = re.search(r'trade rate.+?(\d+(?:\.\d+)?)%', query_lower)
        if trade_rate:
            constraints['trade_rate'] = float(trade_rate.group(1))
        
        if constraints:
            entities['constraints'] = constraints
        
        return entities
    
    def _analyze_intent(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the business intent of the query"""
        query_lower = query.lower()
        
        # Intent scoring
        intent_scores = {}
        
        # Optimization intent
        optimization_keywords = ['optimize', 'best', 'maximize', 'invest', 'spend', 'allocate', 'how to']
        opt_score = sum(1 for kw in optimization_keywords if kw in query_lower)
        if 'budget' in entities or '$' in query:
            opt_score += 2
        intent_scores['optimization'] = opt_score
        
        # Analysis intent
        analysis_keywords = ['show', 'tell', 'analyze', 'performance', 'results', 'what']
        analysis_score = sum(1 for kw in analysis_keywords if kw in query_lower)
        intent_scores['analysis'] = analysis_score
        
        # Prediction intent
        prediction_keywords = ['predict', 'forecast', 'estimate', 'expect', 'will']
        pred_score = sum(1 for kw in prediction_keywords if kw in query_lower)
        intent_scores['prediction'] = pred_score
        
        # Simulation intent
        simulation_keywords = ['what if', 'scenario', 'reduce', 'increase', 'change']
        sim_score = sum(1 for kw in simulation_keywords if kw in query_lower)
        intent_scores['simulation'] = sim_score
        
        # Determine primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if any(intent_scores.values()) else 'unknown'
        
        # Determine business objective
        objective = 'unknown'
        if 'volume' in query_lower or 'lift' in query_lower:
            objective = 'maximize_volume'
        elif 'profit' in query_lower:
            objective = 'maximize_profit'
        elif 'roi' in query_lower:
            objective = 'maximize_roi'
        elif 'reduce' in query_lower and 'trade rate' in query_lower:
            objective = 'reduce_trade_rate'
        
        return {
            'primary_intent': primary_intent,
            'intent_scores': intent_scores,
            'objective': objective,
            'confidence': max(intent_scores.values()) / 5.0 if intent_scores else 0.0
        }
    
    def _identify_data_requirements(self, intent_analysis: Dict, entities: Dict) -> Dict[str, Any]:
        """Identify what data is needed to solve the query"""
        intent = intent_analysis['primary_intent']
        template = self.solution_templates.get(intent, {})
        required_data = template.get('required_data', [])
        
        missing_data = []
        available_data = []
        
        for req in required_data:
            if req in entities or any(req in str(v) for v in entities.values()):
                available_data.append(req)
            else:
                missing_data.append(req)
        
        return {
            'required': required_data,
            'available': available_data,
            'missing': missing_data,
            'completeness': len(available_data) / len(required_data) if required_data else 1.0
        }
    
    def _create_solution_plan(self, intent_analysis: Dict, entities: Dict, data_req: Dict) -> Dict[str, Any]:
        """Create detailed solution plan"""
        intent = intent_analysis['primary_intent']
        template = self.solution_templates.get(intent, {})
        
        plan = {
            'approach': template.get('description', 'Unknown approach'),
            'objective': intent_analysis['objective'],
            'steps': template.get('steps', []),
            'data_sources_needed': [
                'Historical promotion database',
                'ML prediction model (BaseUnitsPredictor)',
                'Business formula engine'
            ],
            'expected_outputs': self._define_expected_outputs(intent, entities),
            'estimated_complexity': self._estimate_complexity(intent_analysis, entities),
            'assumptions': self._list_assumptions(entities)
        }
        
        return plan
    
    def _define_expected_outputs(self, intent: str, entities: Dict) -> List[str]:
        """Define what outputs the user should expect"""
        if intent == 'optimization':
            return [
                'Optimal promotion calendar',
                'Expected volume lift and profit',
                'Investment allocation by product/retailer',
                'Risk assessment and alternatives'
            ]
        elif intent == 'analysis':
            return [
                'Historical performance metrics',
                'Best performing promotions',
                'ROI analysis',
                'Trend insights'
            ]
        else:
            return ['Analysis results', 'Recommendations', 'Key insights']
    
    def _estimate_complexity(self, intent_analysis: Dict, entities: Dict) -> str:
        """Estimate query complexity"""
        score = 0
        score += len(entities.get('products', [])) * 1
        score += len(entities.get('retailers', [])) * 1
        score += 2 if intent_analysis['primary_intent'] == 'optimization' else 1
        
        if score <= 3:
            return 'Simple'
        elif score <= 6:
            return 'Medium'
        else:
            return 'Complex'
    
    def _list_assumptions(self, entities: Dict) -> List[str]:
        """List assumptions being made"""
        assumptions = [
            'Historical data is representative of future performance',
            'Business constraints are hard limits',
            'ML model predictions are reliable'
        ]
        
        if 'budget' in entities:
            assumptions.append('Budget is total available investment')
        
        return assumptions
    
    def _generate_confirmation_request(self, original_query: str, solution_plan: Dict) -> str:
        """Generate human-readable confirmation request"""
        request = f"""
I understand you want to: {solution_plan['approach']}

Based on your query: "{original_query}"

Here's my interpretation and proposed solution:

🎯 **Objective**: {solution_plan['objective']}

📋 **My Plan**:
"""
        
        for i, step in enumerate(solution_plan['steps'], 1):
            request += f"{i}. {step}\n"
        
        request += f"""
📊 **Expected Outputs**:
"""
        for output in solution_plan['expected_outputs']:
            request += f"• {output}\n"
        
        request += f"""
⚠️ **Key Assumptions**:
"""
        for assumption in solution_plan['assumptions']:
            request += f"• {assumption}\n"
        
        request += """
**Is this interpretation correct? Would you like me to:**
- Proceed with this plan?
- Modify any steps?
- Add additional analysis?
- Change the objective?

Please confirm or suggest changes."""
        
        return request
    
    async def _process_user_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Process user's confirmation or modifications"""
        # This would handle user responses and modify the solution plan
        return {"status": "plan_updated", "message": "Solution plan updated based on feedback"}


# Test the enhanced agent
if __name__ == "__main__":
    import asyncio
    
    async def test_enhanced_query_agent():
        agent = TradePromotionQueryAgent("trade_query_agent")
        
        test_queries = [
            "I have $3MM to invest at retailer Walmart across products A, B & C over 4 weeks. I cannot go beyond 30% discount and can promote maximum 2 weeks.",
            "Show me the best promotions for Product A at Target",
            "What if I reduce my trade rate from 20% to 18% at Walmart? How much volume will I lose?",
            "Optimize my promotion spend to maximize profit"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            
            result = await agent.process_request({"query": query})
            
            print(f"Intent: {result['interpretation']['primary_intent']}")
            print(f"Objective: {result['interpretation']['business_objective']}")
            print(f"Entities: {result['interpretation']['entities_found']}")
            print(f"Confidence: {result['interpretation']['confidence']:.2f}")
            print(f"\nConfirmation Request:\n{result['confirmation_request']}")
    
    asyncio.run(test_enhanced_query_agent())