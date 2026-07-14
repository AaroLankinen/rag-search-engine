import argparse
import base64
import mimetypes
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image and Rewrite Query CLI")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--query", required=True, help="Text query to rewrite based on the image")
    
    args = parser.parse_args()
    
    # Guess the MIME type
    mime_type, _ = mimetypes.guess_type(args.image)
    if not mime_type:
        mime_type = "image/jpeg"
        
    # Open the image file as binary and read its contents
    try:
        with open(args.image, "rb") as f:
            image_data = f.read()
    except FileNotFoundError:
        print(f"Error: Image file '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)
        
    # Base64 encode the image data
    base64_image = base64.b64encode(image_data).decode("utf-8")
    
    # Load environment variables
    # Resolve the absolute path to the workspace .env file
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(os.path.dirname(cli_dir), '.env')
    load_dotenv(dotenv_path, override=True)
    
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("Error: OPENROUTER_API_KEY is not set in environment.", file=sys.stderr)
        sys.exit(1)
        
    # Initialize OpenAI client pointed at OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )
    
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": args.query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
        temperature=0.0,
    )
    
    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")

if __name__ == "__main__":
    main()
