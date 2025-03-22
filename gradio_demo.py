import gradio as gr
from gradio_rerun import Rerun

import os
import tempfile
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

from inference import preprocess_point_cloud, generate_layout, create_default_code_template
from visualize import visualize_layout_with_rerun
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose
from spatiallm import Layout


model = None
tokenizer = None
template_file_path = "code_template.txt"
output_dir = "outputs"

def load_model(model_path):
    global model, tokenizer
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if torch.cuda.is_available():
            model.to("cuda")
        else:
            model.to("cpu")
            
        model.set_point_backbone_dtype(torch.float32)
        model.eval()
    return model, tokenizer 


def process_point_cloud(
    point_cloud_file, 
    model_path, 
    custom_prompt=None,
    top_k=10, 
    top_p=0.95, 
    temperature=0.6, 
    num_beams=1,
    max_new_tokens=4096,
    visualization_radius=0.01,
    max_visualization_points=1000000,
    progress=gr.Progress()
):
    
    code_template_file = create_default_code_template()

    # Loading model
    model, tokenizer = load_model(model_path)

    # point cloud preprocessing
    point_cloud = load_o3d_pcd(point_cloud_file.name)
    point_cloud = cleanup_pcd(point_cloud)
    points, colors = get_points_and_colors(point_cloud)
    min_extent = np.min(points, axis=0)
    
    grid_size = Layout.get_grid_size()
    num_bins = Layout.get_num_bins()
    input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)
        
    # Generate layout
    layout, layout_str = generate_layout(
        model,
        input_pcd,
        tokenizer,
        code_template_file,
        custom_prompt,
        top_k,
        top_p,
        temperature,
        num_beams,
        max_new_tokens
    )
    
    # Post-processing layout
    layout.translate(min_extent)
    pred_language_string = layout.to_language_string()
    
    # Save layout to output directory
    layout_path = os.path.join(output_dir, "layouts", "layout.txt")
    with open(layout_path, "w") as f:
        f.write(pred_language_string)
        
    return layout_path, pred_language_string, point_cloud_file.name, visualization_radius, max_visualization_points

def gradio_interface():
    with gr.Blocks(title="SpatialLM Gradio | 3D Scene Understanding with  Large Language Model ") as demo:
        gr.Markdown("# SpatialLM Gradio | 3D Scene Understanding with  Large Language Model ")
        gr.Markdown("SpatialLM: A 3D Large Language Model for Structured Scene Understanding, Processing Point Cloud Data from Monocular Videos, RGBD Images, and LiDAR.")

        gr.Markdown("Upload a point cloud file (.ply)")

        point_cloud_path_state = gr.State()
        radius_state = gr.State()
        max_points_state = gr.State()
        layout_text_state = gr.State()
        
        with gr.Row():
            with gr.Column(scale=1):
                point_cloud_file = gr.File(label="Upload Point Cloud (.ply)")
                
                model_path = gr.Dropdown(
                    label="Model Selection", 
                    choices=[
                        "manycore-research/SpatialLM-Llama-1B",
                        "manycore-research/SpatialLM-Qwen-0.5B"
                    ],
                    value="manycore-research/SpatialLM-Llama-1B"
                )                
                custom_prompt = gr.Textbox(
                    label="Custom Prompt",
                    placeholder="Detect walls, doors, windows, boxes...",
                    value="Detect walls, doors, windows, boxes"
                )
                       
                with gr.Accordion("Generation Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.6, step=0.1,
                        label="Temperature"
                    )
                    num_beams = gr.Slider(
                        minimum=1, maximum=10, value=1, step=1,
                        label="Number of Beams"
                    )
                    max_new_tokens = gr.Slider(
                        minimum=100, maximum=8192, value=4096, step=100,
                        label="Max New Tokens"
                    )
                
                with gr.Accordion("Visualization Settings", open=False):
                    visualization_radius = gr.Slider(
                        minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                        label="Point Radius for Visualization"
                    )
                    max_visualization_points = gr.Slider(
                        minimum=10000, maximum=5000000, value=1000000, step=10000,
                        label="Max Points for Visualization"
                    )
                
                submit_btn = gr.Button("Generate Layout", variant="primary", size="lg")

            
            with gr.Column(scale=1):
                rerun_viewer = Rerun(
                    streaming=True,
                    height=700,
                    panel_states={
                        "time": "collapsed",
                        "blueprint": "hidden",
                        "selection": "hidden",
                    },
                )
        
        process_outputs = submit_btn.click(
            fn=process_point_cloud,
            inputs=[
                point_cloud_file, 
                model_path, 
                custom_prompt,
                top_k, 
                top_p, 
                temperature, 
                num_beams,
                max_new_tokens,
                visualization_radius,
                max_visualization_points
            ],
            outputs=[gr.State(), layout_text_state, point_cloud_path_state, radius_state, max_points_state]
        )
        process_outputs.then(
            fn=visualize_layout_with_rerun,
            inputs=[point_cloud_path_state, layout_text_state, radius_state, max_points_state],
            outputs=[rerun_viewer]
        )
        
    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
