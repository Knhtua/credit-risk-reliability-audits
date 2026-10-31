import json
import pandas as pd
from sklearn.metrics import average_precision_score

submission = pd.read_csv("submission.csv").sort_values("ROW_ID")
answers = pd.read_csv("submission_test.csv").sort_values("ROW_ID")
assert list(submission.columns) == ["ROW_ID", "DOWNGRADE_6M"]
assert len(submission) == len(answers)
assert submission["ROW_ID"].tolist() == answers["ROW_ID"].tolist()
prob = pd.to_numeric(submission["DOWNGRADE_6M"], errors="raise")
assert prob.between(0, 1).all()
score = average_precision_score(answers["DOWNGRADE_6M"], prob)
print(json.dumps({"competition_id": "bond-credit-risk-6m", "score": score}))
