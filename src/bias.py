"""Representation diagnostics using Fairlearn, without protected-group claims."""
import numpy as np
from fairlearn.metrics import MetricFrame, count, selection_rate

def representation(metadata, selected_ids):
    eligible = metadata[metadata.eligible_sales].copy()
    result = {'reference':'491 sales-eligible household series; price coverage confirmed in Module 2',
              'thresholds':{'minimum_selected_store_series':5,'minimum_selected_department_series':10},
              'protected_group_fairness':'Not assessed; customer demographics absent', 'groups':{},'flags':[]}
    chosen = eligible.id.isin(selected_ids).astype(int).to_numpy()
    for column in ['store_id','dept_id']:
        metrics = MetricFrame(metrics={'candidate_count':count,'selection_rate':selection_rate},
                              y_true=np.zeros(len(eligible)),y_pred=chosen,sensitive_features=eligible[column].to_numpy())
        table = metrics.by_group
        records = []
        total = int(chosen.sum())
        for group, row in table.iterrows():
            n = int(row.candidate_count)
            k = int(round(n * row.selection_rate))
            records.append(dict(group=group,candidate_count=n,selected_count=k,
                                candidate_share=n/len(eligible),selected_share=k/total,
                                selection_rate=float(row.selection_rate)))
            limit = 5 if column=='store_id' else 10
            if k<limit:
                result['flags'].append(f'{column} {group}: {k} selected series below review threshold {limit}')
        result['groups'][column] = records
    return result
