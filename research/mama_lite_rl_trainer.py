# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from mama_lite_rl_env import MAMATradingEnv, prepare_env_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("RLTrainer")


def train_mama_rl():
    import torch
    from gnn_model_trainer import SimpleGCN

    logger.info("Preparing data for RL Training...")
    price_df, regime_df, feat_df = prepare_env_data()

    # Load GNN Model and Build Adjacency
    GNN_DATA_DIR = r"d:\gg\data\gnn"
    adj_df = pd.read_csv(
        os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv"), index_col=0
    )
    adj = adj_df.values
    adj = adj + np.eye(adj.shape[0])
    rowsum = adj.sum(1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    adj_tensor = torch.FloatTensor(adj_norm)

    gnn_model = SimpleGCN(in_channels=2, out_channels=1)
    weight_path = os.path.join(GNN_DATA_DIR, "gnn_weights.pth")
    if os.path.exists(weight_path):
        gnn_model.load_state_dict(torch.load(weight_path))
        logger.info("GNN weights loaded for dynamic inference.")
    else:
        logger.warning("GNN weights not found. Using random weights!")

    # Temporal Split: Train (< 2023), Test (>= 2023)
    split_date = "2023-01-01"

    # Training Data
    price_train = price_df.loc[:split_date]
    regime_train = regime_df.loc[:split_date]
    feat_train = feat_df.loc[:split_date]

    # Test Data
    price_test = price_df.loc[split_date:]
    regime_test = regime_df.loc[split_date:]
    feat_test = feat_df.loc[split_date:]

    logger.info(
        f"Training Period: {regime_train.index[0].date()} ~ {regime_train.index[-1].date()}"
    )
    logger.info(
        f"Testing Period: {regime_test.index[0].date()} ~ {regime_test.index[-1].date()}"
    )

    train_env = MAMATradingEnv(
        price_train, regime_train, gnn_model, adj_tensor, feat_train
    )
    test_env = MAMATradingEnv(price_test, regime_test, gnn_model, adj_tensor, feat_test)

    # Define Model: PPO
    model = PPO(
        "MlpPolicy", train_env, verbose=1, learning_rate=0.0003, n_steps=128, gamma=0.99
    )

    logger.info("Starting Training for 30,000 steps...")
    model.learn(total_timesteps=30000)

    # Save Model
    save_path = r"d:\gg\research\mama_lite_ppo_model"
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")

    # Evaluation on Test Data
    obs, info = test_env.reset()
    test_navs = []

    for _ in range(len(regime_test) - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        test_navs.append(info["nav"])
        if done:
            break

    final_test_nav = test_navs[-1] if test_navs else 1.0
    logger.info(f"Out-of-Sample Test Result: Final NAV = {final_test_nav:.4f}")

    # Save a small report
    report_path = r"d:\gg\docs\reports\mama_lite_rl_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MAMA Lite: RL 비중 최적화 성과 보고서 (Refined)\n\n")
        f.write(
            "사용자 피드백과 시스템 분석을 반영하여 시계열 분리 학습 및 거래 비용을 포함한 결과입니다.\n\n"
        )
        f.write("## 1. 학습 및 테스트 요약\n")
        f.write(f"- **총 학습 스텝**: 30,000 steps\n")
        f.write(
            f"- **학습 기간**: {regime_train.index[0].date()} ~ {regime_train.index[-1].date()}\n"
        )
        f.write(
            f"- **테스트 기간 (Out-of-Sample)**: {regime_test.index[0].date()} ~ {regime_test.index[-1].date()}\n"
        )
        f.write(f"- **테스트 최종 NAV**: {final_test_nav:.4f}\n\n")

        f.write("## 2. 주요 개선 사항 반영\n")
        f.write(
            "1. **GNN 실전 연동**: QQQ 단일 지수가 아닌 GNN이 선정한 Top 3 종목의 실시간 수익률을 Alpha 자산으로 반영.\n"
        )
        f.write(
            "2. **거래 비용 현실화**: 비중 변경 시 0.1%의 거래 비용(Transaction Cost)을 보상 함수에 페널티로 부과.\n"
        )
        f.write(
            "3. **과적합 방지**: 2023년 이후 데이터를 학습에서 제외하여 순수한 Out-of-Sample 성능 검증.\n"
        )


if __name__ == "__main__":
    train_mama_rl()
