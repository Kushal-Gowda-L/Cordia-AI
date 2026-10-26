"""Multi-source heart-risk model: one FT-Transformer encoder per public dataset,
combined by late fusion at prediction time.

None of the datasets share patients, so no model is ever trained on rows
stitched together from different datasets. Each encoder learns from its own
dataset's real patients; a patient is described once, as a `profile` (see
profile.py), and every encoder whose inputs can be derived from that profile
scores it. The fusion step then weights each encoder by how good it was on
its own held-out patients and by how much of its input the profile covers.

    python train_multisource.py --data-dir datasets/multisource
    python predict_multisource.py --example
"""
