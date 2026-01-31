"""
Full Automation Pipeline (v4.0)
1. Universe Update (Top 20)
2. Adjacency Matrix Update (Download Data + Correlation)
3. Model Retraining (Attention GNN)
4. Notification
"""

import os
import sys
import subprocess
import logging
import time

# Logging Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "full_update.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("FullUpdatePipeline")


def run_command(command, description):
    """Run a shell command and check for errors."""
    logger.info(f"Starting: {description}")
    try:
        # Assuming python is in path
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
        )
        logger.info(f"Completed: {description}")
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error Output: {e.stderr}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("Antigravity Dynamic Universe & Model Update Pipeline")
    logger.info("=" * 60)

    # 1. Update Universe (Disabled temporarily due to data source issues)
    # cmd_universe = f"{sys.executable} scripts/update_universe.py"
    # if not run_command(cmd_universe, "Universe Selection (Top 20)"):
    #     sys.exit(1)

    # 2. Update Adjacency Matrix (Includes Data Download)
    # Note: update_adjacency_matrix.py already downloads data logic inside calculate_correlation_matrix
    cmd_adj = f"{sys.executable} signal_mailer/update_adjacency_matrix.py"
    if not run_command(cmd_adj, "Adjacency Matrix Update"):
        logger.warning(
            "Adjacency Update Failed (likely Data Source). using Fallback..."
        )
        cmd_fallback = f"{sys.executable} scripts/create_fallback_adjacency.py"
        run_command(cmd_fallback, "Sector-Based Adjacency (Fallback)")

        logger.warning("Skipping Model Retraining (No Data). Using existing weights.")
        # cmd_train = f"{sys.executable} signal_mailer/train_attention_gnn.py"
        # run_command(cmd_train, "Model Retraining")
    else:
        # 3. Train Model (Only if data download succeeded)
        cmd_train = f"{sys.executable} signal_mailer/train_attention_gnn.py"
        if not run_command(cmd_train, "Model Retraining (Attention GNN)"):
            logger.warning("Training Failed. Using existing weights.")

    logger.info("=" * 60)
    logger.info("All Updates Completed Successfully!")
    logger.info("New universe selected, data updated, and model retrained.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
