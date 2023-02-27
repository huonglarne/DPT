# DPT

This repo runs Huggingface's implementation of [DPT](https://huggingface.co/docs/transformers/model_doc/dpt).

The code is borrowed from this [tutorial](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DPT)

# Run with Moreh framework

    conda create -n dpt python=3.8
    conda activate dpt
    update-moreh --force --target 23.3.0
    pip3 install transformers Pillow

# Inference

    python semantic_segmentation.py
    python depth_estimation.py
