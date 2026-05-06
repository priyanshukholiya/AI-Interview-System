import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_system_prompt(resume_text: str = ""):
    base_prompt = """You are an expert technical interviewer. 
    Your goal is to assess the candidate's engineering skills.
    Keep your responses concise (under 2 sentences) and conversational.
    Do not write code blocks, just speak naturally."""
    
    if resume_text:
        # This is the RAG part: Injecting external knowledge into the prompt
        return f"{base_prompt} \n\nCANDIDATE CONTEXT: The candidate has provided a resume with the following details: {resume_text}. Use this information to ask specific, deep-dive technical questions about their experience and projects."
    else:
        return f"{base_prompt} \n\nNo resume provided. Start by asking them to introduce themselves and their tech stack."

async def get_ai_response(transcript: str, conversation_history: list, resume_text: str = ""):
    conversation_history.append({"role": "user", "content": transcript})
    
    # We generate the prompt dynamically based on whether RAG data is present
    dynamic_sys_prompt = generate_system_prompt(resume_text)
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": dynamic_sys_prompt},
            *conversation_history
        ],
        temperature=0.6,
        max_tokens=150,
    )
    
    response_text = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": response_text})
    return response_text, conversation_history

# NEW: The PDF Study Guide Generator Function
async def generate_study_guide(conversation_history: list):
    # If they ended immediately, return a placeholder
    if len(conversation_history) < 2:
        return {"guide": [{"question": "No questions asked.", "ideal_answer": "Complete a longer interview next time!"}]}

    # Convert the history dictionary into a readable text block
    history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history])
    
    prompt = """You are an expert tech recruiter. Review the following interview transcript.
    Extract the main questions the 'ASSISTANT' asked the 'USER'.
    For each question, write a short, crisp, perfect response (2 to 3 sentences maximum) that the candidate should use to study for their real interview.
    You MUST return strictly a JSON object matching this exact schema:
    {"guide": [{"question": "The question asked...", "ideal_answer": "The crisp perfect answer..."}]}
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Transcript:\n{history_text}"}
        ],
        response_format={"type": "json_object"}, # Forces the AI to return clean JSON
        temperature=0.2,
    )
    
    return json.loads(completion.choices[0].message.content)