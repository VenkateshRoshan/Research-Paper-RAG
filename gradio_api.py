import gradio as gr
from google.cloud import aiplatform
from google.auth import credentials, load_credentials_from_dict
import json
from google.protobuf.json_format import MessageToDict
from datetime import datetime

# Load credentials and initialize client
credentials, project_id = load_credentials_from_dict(
    json.load(open('research-paper-rag-0a8819b735b9.json'))
)

client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
client = aiplatform.gapic.PredictionServiceClient(
    client_options=client_options,
    credentials=credentials
)

# Endpoint configuration
project = "research-paper-rag"
location = "us-central1"
endpoint_id = "3729166308927864832"
endpoint = f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"

def format_references(references):
    """Format references into a readable string"""
    formatted_text = ""
    for i, ref in enumerate(references, 1):
        ref_dict = dict(ref)
        formatted_text += f"\n[{i}] Title: {ref_dict.get('title', '')}\n"
        formatted_text += f"    Authors: {', '.join(ref_dict.get('authors', []))}\n"
        formatted_text += f"    Categories: {ref_dict.get('categories', '')}\n"
        formatted_text += f"    Relevance Score: {ref_dict.get('relevance_score', 0.0):.2f}\n"
        formatted_text += f"    Citation: {ref_dict.get('citation', '')}\n\n"
    return formatted_text

def predict_query(query, max_tokens):
    try:
        # Prepare input data
        input_data = {
            "instances": [{
                "query": query,
                'max_tokens': max_tokens,
                'num_papers': 2
            }]
        }
        
        # Get prediction
        response = client.predict(endpoint=endpoint, instances=input_data["instances"])
        prediction = dict(response.predictions[0])
        
        # Format response
        main_response = prediction['response']
        references = format_references(prediction['references'])
        
        return main_response, references
    
    except Exception as e:
        return f"Error: {str(e)}", "Error retrieving references"

# Create Gradio interface
with gr.Blocks(title="Research Paper RAG System") as demo:
    gr.Markdown("# Research Paper RAG Query System 📚 (Implemented on Google/Flan-T5-Based Model )")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Enter your query",
                placeholder="What are the recent advances in transformer architecture?",
                lines=3
            )
            token_slider = gr.Slider(
                minimum=50,
                maximum=300,
                value=150,
                step=10,
                label="Max Tokens"
            )
            submit_btn = gr.Button("Submit Query", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Generated Response")
            response_output = gr.Textbox(
                label="Response",
                lines=10,
                show_copy_button=True
            )
        
        with gr.Column():
            gr.Markdown("### Reference Papers")
            references_output = gr.Textbox(
                label="References",
                lines=10,
                show_copy_button=True
            )
    
    # Handle submission
    submit_btn.click(
        fn=predict_query,
        inputs=[query_input, token_slider],
        outputs=[response_output, references_output]
    )
    
    gr.Markdown("""
    ### Usage Notes:
    - Enter your research-related query in the text box
    - Adjust the max tokens slider to control response length (50-300)
    - Click Submit to get AI-generated response with relevant paper references
    - Use copy buttons to easily copy response or references
    """)

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)