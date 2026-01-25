# -*- coding: utf-8 -*-
"""
The Council: AI-Driven Risk Management Module
Uses Google Gemini API to act as a qualitative risk committee (Prosecutor, Defender, Judge).
"""

import os
import json
import logging
import google.generativeai as genai
from dataclasses import dataclass

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TheCouncil")


@dataclass
class CouncilVerdict:
    discount_factor: float
    reason: str
    prosecutor_arg: str
    defender_arg: str


class DebateCouncil:
    def __init__(self, api_key):
        """
        Initialize The Council with Gemini API Key
        """
        if not api_key:
            logger.warning(
                "No API Key provided for The Council. AI Risk Module disabled."
            )
            self.model = None
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                "gemini-1.5-flash"
            )  # Use Flash for speed/cost
            self.api_ready = True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.api_ready = False

    def convene_council(self, market_data: dict, news_context: str) -> CouncilVerdict:
        """
        Convenes the Council to determine the Discount Factor.

        Args:
            market_data (dict): Numerical market metrics (VIX, Scores, Prices).
            news_context (str): Textual news summaries or headlines.

        Returns:
            CouncilVerdict: Object containing discount factor and reasons.
        """
        # 1. Circuit Breaker (Cost/Speed Optimization)
        # Skip if market is calm (VIX < 20) AND Quant Score is Healthy (> 70)
        # However, if 'news_context' contains alarming keywords, we might force it.
        # For now, we follow the strict rule to save costs unless news is very negative.
        vix = market_data.get("vix", 0)
        quant_score = market_data.get("quant_score", 0)

        # Force convene if VIX is high or Score is low, OR if it's explicitly requested (e.g. testing)
        # Here we implement the "Skip" logic
        if vix < 20 and quant_score > 70:
            return CouncilVerdict(
                discount_factor=1.0,
                reason="Market is calm (VIX < 20, Score > 70). The Council remains in recess.",
                prosecutor_arg="N/A",
                defender_arg="N/A",
            )

        if not self.api_ready:
            return CouncilVerdict(1.0, "AI Module Unavailable", "", "")

        # 2. Prepare The Brief (Prompt)
        prompt = f"""
        You are 'The Council', an AI Risk Management Committee for a quantitative hedge fund. 
        Your goal is to screen for "Black Swan" or "Tail Risks" that numerical models might miss.

        [MARKET DATA]
        {json.dumps(market_data, indent=2)}

        [NEWS CONTEXT]
        {news_context}

        [INSTRUCTIONS]
        Adopt three personas sequentially to analyze the situation. 
        **IMPORTANT: All reasoning and output MUST be in Korean (한국어).**

        1. PROSECUTOR (검사): Identify potential tail risks, negative news, or structural weaknesses. Be pessimistic and critical.
           - Focus on: "Why might the market crash tomorrow?"
        2. DEFENDER (변호사): Identify bullish factors, resilience, or positive catalysts. Be optimistic and rational.
           - Focus on: "Why is the trend still valid?"
        3. JUDGE (판사): Weigh both arguments and decide a 'discount_factor' (0.5 to 1.0) to apply to the portfolio exposure.
           - Provide a "verdict_reason" that explains the decision clearly to a human investor.
           - 1.0: "Trust the Quant Signal completely." (No external threat)
           - 0.8: "Minor Caution." (Some noise, tighten stops)
           - 0.5: "Severe Danger." (War, Pandemic, Systemic Crash imminent)

        [OUTPUT FORMAT]
        Return ONLY a JSON object with no markdown formatting:
        {{
            "prosecutor_arg": "핵심 위험 요인 (한글)...",
            "defender_arg": "상승 지지 요인 (한글)...",
            "verdict_reason": "최종 판결 및 조언 (한글)...",
            "discount_factor": 0.xx
        }}
        """

        try:
            # 3. Deliberation
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Low volatility for consistent judging
                    response_mime_type="application/json",
                ),
            )

            # 4. Verdict extraction
            result_text = response.text.strip()
            # Handle potential markdown wrappers if model ignores instruction
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]

            data = json.loads(result_text)

            # Safety checks
            discount = float(data.get("discount_factor", 1.0))
            discount = max(0.0, min(1.0, discount))  # Clamp between 0 and 1

            return CouncilVerdict(
                discount_factor=discount,
                reason=data.get("verdict_reason", "No reason provided."),
                prosecutor_arg=data.get("prosecutor_arg", ""),
                defender_arg=data.get("defender_arg", ""),
            )

        except Exception as e:
            logger.error(f"Council Deliberation Failed: {e}")
            # Fail-safe: Do not punish portfolio on AI error.
            return CouncilVerdict(1.0, f"Council Error: {str(e)}", "", "")


# Example Usage Stub
if __name__ == "__main__":
    # Test with dummy key (Replace with real one for actual test)
    # council = DebateCouncil(api_key="TEST_KEY")
    pass
