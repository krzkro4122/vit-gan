from src.v2.training import train_model
from ray import tune
import os
import ray


if __name__ == "__main__":
    # train_loader()

    os.environ["RAY_TMPDIR"] = "/tmp/ray_tmp"
    ray.init(num_gpus=1)
    search_space = {
        "generator_learning_rate": tune.loguniform(1e-6, 1e-4),
        "discriminator_learning_rate": tune.loguniform(1e-6, 1e-4),
        "embed_dim": tune.choice([128, 256, 512]),
        "num_heads": tune.choice([4, 8]),
        "batch_size": tune.choice([128, 256]),
    }

    analysis = tune.run(
        train_model,
        resources_per_trial={"gpu": 1},
        config=search_space,
        num_samples=10,
        metric="fid_score",
        mode="min",
    )

    print("Best config: ", analysis.best_config)
