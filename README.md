# Disease Test Model Evaluation

## Model Performance Metrics

### Accuracy: 87%
**Definition:** Percentage of all patients diagnosed correctly

**Interpretation:** Very good doctor! The model correctly identifies the health status (sick or healthy) for 87% of all patients.

---

### Precision: 73%
**Definition:** When test says 'sick', how often is it right?

**Interpretation:** Some false alarms, but okay. When the model predicts a patient is sick, it's correct 73% of the time. This means about 27% are false positives.

---

### Recall (Sensitivity): 86%
**Definition:** Of all sick people, how many did the test catch?

**Interpretation:** Catches most sick people! ✅

The model successfully identifies 86% of all patients who are actually sick. Only 14% of sick patients are missed (false negatives).

---

### AUC (Area Under Curve): 93%
**Definition:** Overall ability to distinguish sick from healthy

**Interpretation:** Excellent diagnostic test! ✅✅

An AUC of 93% indicates the model has excellent discriminative ability - it's very good at separating sick patients from healthy ones across all threshold settings.

---

## Summary

This disease test model shows strong overall performance with:
- High accuracy (87%)
- Excellent recall - catching most sick patients (86%)
- Outstanding AUC score (93%)
- Acceptable precision with some room for improvement (73%)

The model is particularly good at not missing sick patients, which is often the most critical factor in medical diagnostics.