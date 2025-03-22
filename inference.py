import os
import tempfile
import numpy as np
import torch
import gradio as gr
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose


def preprocess_point_cloud(points, colors, grid_size, num_bins):
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def generate_layout(
    model,
    point_cloud,
    tokenizer,
    code_template,
    custom_prompt=None,
    top_k=10,
    top_p=0.95,
    temperature=0.6,
    num_beams=1,
    max_new_tokens=4096,
):
    if custom_prompt:
        prompt = f"<|point_start|><|point_pad|><|point_end|>{custom_prompt}. The reference code is as followed: {code_template}"
    else:
        prompt = f"<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {code_template}"

    # prepare the conversation data
    if hasattr(model.config, "model_type"):
        if model.config.model_type == "spatial_llama":
            conversation = [{"role": "user", "content": prompt}]
        elif model.config.model_type == "spatial_qwen":
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        else:
            # Fallback for other model types
            conversation = [{"role": "user", "content": prompt}]
    else:
        # Default fallback
        conversation = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        input_ids=input_ids, 
        point_clouds=point_cloud,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    generate_texts = []
    for text in streamer:
        generate_texts.append(text)
    
    layout_str = "".join(generate_texts)
    layout = Layout(layout_str)
    layout.undiscretize_and_unnormalize()
    return layout, layout_str


def create_default_code_template():
    template = """
    # Define the layout
    layout = []

    # Add walls
    layout.append({
        "class": "wall",
        "label": "wall",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "length": 4.0,
        "width": 0.2,
        "height": 2.5,
        "rotation_angle": 0.0
    })

    # Add doors
    layout.append({
        "class": "door",
        "label": "door",
        "x": 1.5,
        "y": 0.0,
        "z": 0.0,
        "length": 1.0,
        "width": 0.1,
        "height": 2.1,
        "rotation_angle": 0.0
    })

    # Add windows
    layout.append({
        "class": "window",
        "label": "window",
        "x": 2.5,
        "y": 0.0,
        "z": 1.0,
        "length": 1.2,
        "width": 0.1,
        "height": 1.0,
        "rotation_angle": 0.0
    })
    """
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    temp_file.write(template)
    temp_file.close()
    return temp_file.name
