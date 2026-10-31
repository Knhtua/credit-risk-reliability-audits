from pathlib import Path
import pandas as pd

submission = pd.read_csv("submission.csv")
answers = pd.read_csv("submission_test.csv")
assert list(submission.columns) == ["ROW_ID", "DOWNGRADE_6M"]
assert len(submission) == len(answers)
assert submission["ROW_ID"].is_unique
assert set(submission["ROW_ID"]) == set(answers["ROW_ID"])
prob = pd.to_numeric(submission["DOWNGRADE_6M"], errors="raise")
assert prob.between(0, 1).all()
print("submission format is valid")
