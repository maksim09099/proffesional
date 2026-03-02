from src.data.data_audit import run_data_audit
from src.data.preprocessing import build_processed_dataset
from src.models.training import train_model
from src.models.evaluation import evaluate_model


if __name__ == "__main__":
    print("Step 1: data audit")
    print(run_data_audit())

    print("Step 2: preprocessing")
    print(build_processed_dataset())

    print("Step 3: training")
    print(train_model())

    print("Step 4: evaluation")
    print(evaluate_model())
