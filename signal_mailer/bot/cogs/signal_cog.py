# -*- coding: utf-8 -*-
import discord
from discord import app_commands
from discord.ext import commands
from signal_mailer.signal_detector import SignalDetector
import logging

logger = logging.getLogger("SignalCog")


class SignalCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.detector = SignalDetector()  # Initialize once

    @app_commands.command(
        name="signal", description="Check current Antigravity market signal"
    )
    async def signal(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()  # Acknowledge interaction (avoids timeout)

        logger.info(f"Signal command triggered by {interaction.user}")

        try:
            # Run detection logic
            # report = await self.bot.loop.run_in_executor(None, self.detector.detect, False)
            # SignalDetector is not async, running it directly for now as it's quick enough
            report = self.detector.detect(verbose=False)

            signal = report.get("signal", "UNKNOWN")

            # Color Logic
            color = discord.Color.green()
            if signal == "DANGER":
                color = discord.Color.red()
            elif signal == "WARNING":
                color = discord.Color.gold()  # Yellow/Orange

            # Embed
            embed = discord.Embed(
                title=f"🚀 Real-time Signal: {signal}",
                description=report.get("action_plan", "No plan."),
                color=color,
            )

            # Fields
            embed.add_field(
                name="QQQ Price",
                value=f"${report.get('qqq_price', 0):.2f}",
                inline=True,
            )
            embed.add_field(
                name="VIX", value=f"{report.get('vix', 'N/A')}", inline=True
            )

            # Sniper
            sniper_val = report.get("sniper_signal")
            sniper_str = (
                sniper_val.current_state
                if sniper_val and hasattr(sniper_val, "current_state")
                else "N/A"
            )
            embed.add_field(name="Index Sniper", value=sniper_str, inline=False)

            embed.set_footer(
                text=f"Requested by {interaction.user.display_name} • Antigravity v4.1"
            )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in signal command: {e}")
            await interaction.followup.send(
                f"❌ Error executing signal check: {e}", ephemeral=True
            )

    @app_commands.command(name="ping", description="Check bot latency")
    async def ping(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            f"Pong! 🏓 Latency: {round(self.bot.latency * 1000)}ms"
        )

    @app_commands.command(
        name="scan", description="Scan KR Market for Hybrid Alpha candidates"
    )
    async def scan(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        logger.info(f"Scan command triggered by {interaction.user}")

        try:
            # 1. Load Config (Dynamic to catch KIS changes)
            import yaml
            import os

            config_path = os.path.join(os.getcwd(), "signal_mailer", "config.yaml")
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            if "kis" not in config:
                await interaction.followup.send(
                    "⚠️ KIS API configuration is missing in config.yaml."
                )
                return

            from signal_mailer.kis_api_wrapper import KISAPIWrapper
            from signal_mailer.kr_stock_scanner import KRStockScanner

            # 2. Initialize Scanner
            kis_wrapper = KISAPIWrapper(config["kis"])
            scanner = KRStockScanner(kis_wrapper)

            # 3. Run Scan
            logger.info(f"Cog triggering scan with limit 200...")
            candidates = await self.bot.loop.run_in_executor(
                None, scanner.scan_full_market, 200
            )
            logger.info(f"Scan finished. Found {len(candidates)} candidates.")

            if not candidates:
                logger.warning("No candidates found in scan log check.")
                await interaction.followup.send(
                    "🔍 Scan complete. No candidates found today."
                )
                return

            # 4. Build Result Embed
            embed = discord.Embed(
                title=f"💎 Hybrid Alpha Scan (KR Top 10)",
                description="Logic: `(Close > SMA_5) AND (ROC_1 > 0)`\n*Scan Criteria: Trading Volume Top 200*",
                color=discord.Color.blue(),
            )

            cand_str = ""
            for i, c in enumerate(candidates[:10], 1):
                price = c.get("price", 0)
                roc = c.get("roc_1", 0) * 100
                dist = c.get("dist_sma", 0) * 100
                cand_str += f"{i}. **{c.get('name', 'N/A')}** ({c.get('ticker', 'N/A')})\n   └ {price:,}원 (+{roc:.1f}%) | 5일선 이격: {dist:+.1f}%\n"

            embed.add_field(
                name="Selected Candidates",
                value=cand_str if cand_str else "N/A",
                inline=False,
            )

            embed.set_footer(
                text=f"Fetched by {interaction.user.display_name} • KIS API Live"
            )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in scan command: {e}")
            await interaction.followup.send(f"❌ Scan failed: {e}", ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(SignalCog(bot))
