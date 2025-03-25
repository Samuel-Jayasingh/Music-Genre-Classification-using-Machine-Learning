import gradio as gr
from src.recommender import MusicRecommender

# Instantiate the recommender system
recommender = MusicRecommender()

def get_recommendations(song_name, top_n, method, alpha):
    if not song_name.strip():
        return "Please enter a valid song name."
    recs = recommender.recommend(song_name, top_n=top_n, method=method, alpha=alpha)
    if recs.empty:
        return "No recommendations found. Check the song name or try a different method."
    return recs

# Create a Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("## Music Recommendation System")
    
    with gr.Row():
        song_input = gr.Textbox(label="Enter Song Name", placeholder="e.g., Shape of You")
        top_n_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of Recommendations")
    
    method_radio = gr.Radio(choices=["traditional", "cluster", "llm", "hybrid"],
                             value="hybrid", label="Recommendation Method")
    alpha_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.5, label="Alpha (weight)")
    
    submit_btn = gr.Button("Get Recommendations")
    output_table = gr.DataFrame(label="Recommended Songs")
    
    submit_btn.click(fn=get_recommendations, 
                     inputs=[song_input, top_n_slider, method_radio, alpha_slider],
                     outputs=output_table)

# Launch the Gradio interface on a separate port
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
