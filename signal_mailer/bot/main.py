# -*- coding: utf-8 -*-
import discord
from discord.ext import commands
import yaml
import os
import logging
from typing import Dict, Any

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AntigravityBot")


def load_config() -> Dict[str, Any]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return {}


class AntigravityBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        # Load Cogs
        initial_extensions = ["signal_mailer.bot.cogs.signal_cog"]

        for extension in initial_extensions:
            try:
                await self.load_extension(extension)
                logger.info(f"Loaded extension: {extension}")
            except Exception as e:
                logger.exception(f"Failed to load extension {extension}")

        # Sync Commands
        logger.info("Syncing commands...")
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} commands.")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")


def run_bot() -> None:
    config = load_config()
    discord_cfg = config.get("discord", {})
    token = discord_cfg.get("token")

    if not token or "YOUR_BOT_TOKEN" in token:
        logger.error("Bot Token is missing in config.yaml")
        return

    bot = AntigravityBot()
    bot.run(token)


if __name__ == "__main__":
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.append(project_root)

    run_bot()
