import pyttsx3
from transformers import pipeline

# --- 1. Initialize the Text-to-Speech (TTS) Engine ---
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 155)
except Exception as e:
    print(f"Dynamic Responder: Failed to initialize TTS engine. Error: {e}")
    tts_engine = None

# --- 2. Load the Text Generation Model (distilgpt2) ---
# This model will WRITE new sentences instead of picking from a list.
# It will be downloaded the first time the program runs.
print("Dynamic Responder: Loading text generation model (distilgpt2)...")
try:
    generator = pipeline('text-generation', model='distilgpt2')
    print("Dynamic Responder: Text generation model loaded.")
except Exception as e:
    print(f"Dynamic Responder: Failed to load generator model. Error: {e}")
    generator = None

def generate_dynamic_reply(is_sarcastic, emotion, user_text):
    """
    Generates a dynamic, situational reply using a generative AI model.
    """
    if not generator:
        reply_text = "My creative circuits are down. Can't think of a reply."
    elif is_sarcastic:
        # This is "prompt engineering": we give the AI a role and context.
        prompt = f"""You are a witty, sarcastic AI. A user, who sounds {emotion.lower()}, just said something sarcastic: "{user_text}"
        
        Write a short, clever, and sarcastic comeback. Do not be helpful.

        Your sarcastic reply:"""

        # Generate the text based on the prompt
        # FIXED: Replaced max_length with max_new_tokens to prevent the error.
        generated_outputs = generator(prompt, max_new_tokens=40, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
        generated_text = generated_outputs[0]['generated_text']
        
        # Clean the output to only get the AI's reply part
        reply_text = generated_text.split("Your sarcastic reply:")[1].strip()
        # Further cleaning to remove any lingering prompt artifacts
        reply_text = reply_text.replace(prompt, "").strip()

    else:
        # If the user was not sarcastic, generate a simple, polite acknowledgement.
        # UPDATED PROMPT: This is a much clearer instruction for the AI to prevent nonsensical replies.
        prompt = f"""You are a helpful and friendly AI assistant. A user just said: "{user_text}"
        
        Your task is to provide a short, polite, and conversational acknowledgement. If they ask how you are, say you are doing well.
        
        Here are some examples of good replies:
        - "Thanks for letting me know."
        - "I'm doing well, thanks for asking!"
        - "Okay, got it."

        Now, write a similar polite reply to the user's statement.

        Your polite reply:"""
        
        # FIXED: Replaced max_length with max_new_tokens to prevent the error.
        generated_outputs = generator(prompt, max_new_tokens=30, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
        generated_text = generated_outputs[0]['generated_text']

        reply_text = generated_text.split("Your polite reply:")[1].strip()
        reply_text = reply_text.replace(prompt, "").strip()

    print(f"--> AI Reply: \"{reply_text}\"")

    # Speak the generated reply
    if tts_engine:
        tts_engine.say(reply_text)
        tts_engine.runAndWait()
    else:
        print("--> (TTS engine not available to speak reply)")