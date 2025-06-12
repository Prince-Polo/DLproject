import gradio as gr
from utils_config import generate_cfg_dict

def generate_config_ui(
    mode, config, experiment, seed, semantic_concept, prompt_suffix,
    batch_size, use_wandb, wandb_user, color, color_prompt,
    font, word, optimized_letter, svg_path, jpg_path, png_path
):
    with open('TOKEN', 'r') as f:
        token = f.read().strip()

    args_dict = dict(
        mode=mode,
        config=config,
        experiment=experiment,
        seed=seed,
        semantic_concept=semantic_concept,
        prompt_suffix=prompt_suffix,
        batch_size=batch_size,
        use_wandb=use_wandb,
        wandb_user=wandb_user,
        color=color,
        color_prompt=color_prompt,
        font=font,
        word=word,
        optimized_letter=optimized_letter,
        svg_path=svg_path,
        jpg_path=jpg_path,
        png_path=png_path,
        log_dir="output",
        token=token,
    )

    try:
        cfg = generate_cfg_dict(args_dict)
        return yaml.dump(cfg, sort_keys=False)
    except Exception as e:
        return f"❌ Error: {str(e)}"

iface = gr.Interface(
    fn=generate_config_ui,
    inputs=[
        gr.Dropdown(choices=["word", "svg", "jpg", "png"], label="Mode"),
        gr.Textbox(value="code/config/base.yaml", label="Config Path"),
        gr.Textbox(value="test_experiment", label="Experiment Name"),
        gr.Number(value=0, label="Seed"),
        gr.Textbox(label="Semantic Concept"),
        gr.Textbox(value="minimal flat 2d vector. lineal color. trending on artstation", label="Prompt Suffix"),
        gr.Number(value=1, label="Batch Size"),
        gr.Checkbox(label="Use WandB"),
        gr.Textbox(label="WandB User"),
        gr.Checkbox(label="Use Color"),
        gr.Textbox(label="Color Prompt"),
        gr.Textbox(label="Font (for word mode)"),
        gr.Textbox(label="Word (for word mode)"),
        gr.Textbox(label="Optimized Letter (for word mode)"),
        gr.Textbox(label="SVG Path (for svg mode)"),
        gr.Textbox(label="JPG Path (for jpg mode)"),
        gr.Textbox(label="PNG Path (for png mode)"),
    ],
    outputs="text",
    title="Config Generator for Word-as-Image",
)

if __name__ == "__main__":
    iface.launch()
