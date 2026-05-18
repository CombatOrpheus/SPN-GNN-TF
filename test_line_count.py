from src.spn_gnn_performance.tf_dataset import _fast_line_count
print(_fast_line_count('tests/sample.jsonl'))
print(len(open('tests/sample.jsonl').readlines()))
