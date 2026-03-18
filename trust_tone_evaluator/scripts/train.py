#!/usr/bin/env python
"""Training script for the trust tone evaluator."""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from trust_tone_evaluator.data.ravdess_dataset import create_ravdess_dataloaders
from trust_tone_evaluator.features.opensmile_extractor import OpenSMILEExtractor
from trust_tone_evaluator.models.mlp_model import create_mlp_model
from trust_tone_evaluator.training.trainer import train_model


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train trust tone evaluator")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model_preset", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--feature_set", type=str, default="egemaps", choices=["egemaps", "gemaps"])
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu (default: auto)")

    args = parser.parse_args()

    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

    config.update(
        {
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "checkpoint_dir": args.checkpoint_dir,
            "level_weight": 1.5,
            "score_weight": 1.0,
            "indicator_weight": 0.5,
            "use_focal_loss": False,
            "use_ordinal_loss": False,
            "label_smoothing": 0.1,
            "scheduler": "plateau",
            "scheduler_patience": 5,
            "early_stopping_patience": 20,
        }
    )

    # Initialize feature extractor with functionals (utterance-level)
    logger.info(f"Initializing OpenSMILE extractor ({args.feature_set}, functionals)")
    try:
        feature_extractor = OpenSMILEExtractor(
            feature_set=args.feature_set, feature_level="functionals"
        )
        input_dim = feature_extractor.get_feature_dim()
        logger.info(f"Feature dimension: {input_dim}")
    except ImportError:
        logger.warning(
            "OpenSMILE not available. Install with: pip install opensmile"
        )
        logger.info("Using raw audio features instead")
        feature_extractor = None
        input_dim = 88

    logger.info(f"Loading data from {args.data_dir}")
    train_loader, val_loader, test_loader = create_ravdess_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        feature_extractor=feature_extractor.extract if feature_extractor else None,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Compute class weights for imbalanced trust levels
    class_weights = train_loader.dataset.get_class_weights()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config["class_weights"] = class_weights.to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    logger.info(f"Trust level distribution: {train_loader.dataset.get_trust_distribution()}")
    logger.info(f"Emotion distribution: {train_loader.dataset.get_emotion_distribution()}")

    logger.info(f"Creating MLP model ({args.model_preset} preset)")
    model = create_mlp_model(
        input_dim=input_dim,
        preset=args.model_preset,
    )
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    # Train
    logger.info("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        **config,
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    from trust_tone_evaluator.evaluation.metrics import evaluate_model, TrustMetricsCalculator

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model, test_loader, device=device)

    # Print results
    calculator = TrustMetricsCalculator()
    print("\n" + calculator.summarize(metrics, show_indicators=True))

    # Save final metrics
    metrics_path = Path(args.checkpoint_dir) / "test_metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)
    logger.info(f"Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
