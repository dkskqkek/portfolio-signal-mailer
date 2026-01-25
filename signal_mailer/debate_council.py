import google.generativeai as genai
import os
import json
from datetime import datetime


class DebateCouncil:
    """
    The Council: A Qualitative Risk Assessment Module.
    Role: Analyzes market context (Headlines, Macro) using LLM to provide a "Discount Factor".
    """

    def __init__(self, api_key=None):
        self.api_key = api_key
        if not self.api_key:
            # Try env var or hardcoded fallback (Memory)
            # CAUTION: In production, rely on ENV.
            self.api_key = os.getenv(
                "GEMINI_API_KEY", "AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA"
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-pro")

    def convene_council(self, market_data, news_headlines):
        """
        Convening the Council of Risk.
        Args:
            market_data (dict): key metrics (VIX, RSI, etc)
            news_headlines (list): significant news strings

        Returns:
            discount_factor (float): 0.5 ~ 1.0 (Multiplier for exposure)
            verdict (str): Summary of the debate.
        """

        # 1. Guard: Skip if VIX is low (Save Tokens/Cost)
        vix = market_data.get("VIX", 15.0)
        if vix < 18.0:
            return 1.0, "System Normal (VIX Low). Council Adjourned."

        # 2. Prompt Engineering (The 3-Persona Debate)
        context = f"""
        Market Context: {json.dumps(market_data)}
        News Headlines: {news_headlines}
        Date: {datetime.now().strftime("%Y-%m-%d")}
        """

        prompt = f"""
        Act as 'The Risk Council' for a leveraged equity portfolio.
        You have 3 personas:
        1. PROSECUTOR (Bear): Highlight risks, black swans, and reasons to crash.
        2. DEFENDER (Bull): Highlight resilience, earnings support, and trend continuation.
        3. JUDGE (Final Verdict): Weigh arguments and issue a 'Discount Factor' (0.5 to 1.0).
        
        - 1.0 = Max Exposure (Bullish/Safe)
        - 0.5 = Min Exposure (High Risk/Uncertainty)
        
        {context}
        
        Output format (JSON):
        {{
            "prosecutor_arg": "...",
            "defender_arg": "...",
            "judge_verdict": "...",
            "discount_factor": 0.8
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            # Simple extract JSON
            txt = response.text
            # Cleanup markdown
            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0]
            elif "```" in txt:
                txt = txt.split("```")[1].split("```")[0]

            result = json.loads(txt.strip())

            return float(result["discount_factor"]), result["judge_verdict"]

        except Exception as e:
            print(f"[Council Error] {e}")
            return 1.0, "Council Error. Defaulting to 1.0"


if __name__ == "__main__":
    # Test
    council = DebateCouncil()
    factor, reason = council.convene_council(
        {"VIX": 25.0, "QQQ_RSI": 30},
        ["Fed Signals Rate Hike", "Geopolitical Tension in Middle East"],
    )
    print(f"Factor: {factor}")
    print(f"Verdict: {reason}")
