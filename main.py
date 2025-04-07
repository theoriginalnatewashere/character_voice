import streamlit as st
from elevenlabs.client import ElevenLabs
from io import BytesIO
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

# Set up page config
st.set_page_config(
    page_title="AI Character Voice Chat",
    page_icon="🎭",
    layout="centered"
)


client = ElevenLabs(
    api_key=st.secrets["ELEVENLABS_API_KEY"]
)


CHARACTER_PROFILES = {
    "Sherlock Holmes": {
        "system_prompt": (
            "You are Sherlock Holmes, the famous detective from Arthur Conan Doyle's stories. "
            "You speak with precise, Victorian-era English and exceptional deductive reasoning. "
            "You notice small details and make brilliant deductions from them. "
            "You're confident, sometimes arrogant, and impatient with those who can't follow your reasoning. "
            "You often say phrases like 'Elementary, my dear Watson,' 'The game is afoot,' and "
            "'When you have eliminated the impossible, whatever remains, however improbable, must be the truth.'"
        ),
        "default_voice": "Antoni"  
    },
    "Wednesday Addams": {
        "system_prompt": (
            "You are Wednesday Addams from the Addams Family. You speak with a deadpan, monotone delivery. "
            "Your humor is extremely dark and macabre. You're intelligent and perceptive but "
            "socially detached and cynical. You find joy in the morbid and have zero interest in "
            "normal social pleasantries. Your responses are brief, sardonic, and often disturbing."
        ),
        "default_voice": "Rachel"  
    },
    "Tony Stark": {
        "system_prompt": (
            "You are Tony Stark, also known as Iron Man. You speak with quick wit, technological brilliance, "
            "and occasional narcissism. Use modern slang, tech jargon, and sarcastic humor. "
            "You're confident to the point of arrogance but ultimately heroic. "
            "Make references to your suits, tech innovations, and saving the world. "
            "Occasionally throw in trademark phrases like 'Genius, billionaire, playboy, philanthropist.'"
        ),
        "default_voice": "Josh"  
    }
}

def text_to_speech(text, voice_id, model_id="eleven_multilingual_v2", stability=0.5, similarity_boost=0.5):
    try:
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            voice_settings={
                "stability": stability,
                "similarity_boost": similarity_boost
            },
            output_format="mp3_44100_128"
        )
        
        # Convert the stream to bytes
        if hasattr(audio_stream, 'read'):
            # If it's already a file-like object
            return audio_stream
        else:
            # If it's a generator, consume it into a BytesIO
            buffer = BytesIO()
            for chunk in audio_stream:
                buffer.write(chunk)
            buffer.seek(0)
            return buffer
            
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None


def get_voices():
    try:
        return client.voices.get_all()
    except Exception as e:
        st.error(f"Error fetching voices: {str(e)}")
        return []


def get_character_response(prompt, character):
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        model = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=groq_api_key))

        agent = Agent(
            model,  
            system_prompt=CHARACTER_PROFILES[character]["system_prompt"],  
        )
        
        response = agent.run_sync(prompt).data
        return response
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


if "histories" not in st.session_state:
    st.session_state.histories = {character: [] for character in CHARACTER_PROFILES}

if "current_character" not in st.session_state:
    st.session_state.current_character = list(CHARACTER_PROFILES.keys())[0]


st.title("🎭 AI Character Voice Chat")
st.caption("Chat with famous characters using AI text and voice!")


with st.sidebar:
    st.header("Character Selection")
    new_character = st.radio(
        "Select your character:",
        list(CHARACTER_PROFILES.keys()),
        index=0,
        key="character_selector"
    )
    
  
    if new_character != st.session_state.current_character:
        st.session_state.current_character = new_character
        st.rerun()
    
    st.divider()
    
    
    st.header("Voice Settings")
    voices = get_voices()
    voice_names = [voice.name for voice in voices.voices] if voices else []
    
    if voice_names:
        default_voice = CHARACTER_PROFILES[st.session_state.current_character]["default_voice"]
        default_index = voice_names.index(default_voice) if default_voice in voice_names else 0
        selected_voice = st.selectbox("Select Voice", voice_names, index=default_index)
        voice_id = [voice.voice_id for voice in voices.voices if voice.name == selected_voice][0]
        

        st.caption("Voice Properties:")
        stability = st.slider("Stability", 0.0, 1.0, 0.5, 0.05)
        similarity_boost = st.slider("Similarity Boost", 0.0, 1.0, 0.5, 0.05)
    else:
        st.warning("Could not load voices. Check your ElevenLabs API key.")
        voice_id = None
    
    st.divider()
    st.caption("Made with Streamlit, ElevenLabs, and Groq AI")


avatar_mapping = {
    "user": "👤",
    "Sherlock Holmes": "🕵️",
    "Wednesday Addams": "👧",
    "Tony Stark": "🦸"
}

current_history = st.session_state.histories[st.session_state.current_character]


st.header(f"Talking with {st.session_state.current_character}", divider="rainbow")

for message in current_history:
    role = message["role"]
    avatar = avatar_mapping["user"] if role == "user" else avatar_mapping[st.session_state.current_character]
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        if "audio" in message and message["audio"]:
            st.audio(message["audio"], format="audio/mpeg")


if prompt := st.chat_input(f"Chat with {st.session_state.current_character}..."):

    current_history.append({"role": "user", "content": prompt})
    

    with st.chat_message("user", avatar=avatar_mapping["user"]):
        st.markdown(prompt)

    with st.spinner(f"{st.session_state.current_character} is thinking..."):
        response_text = get_character_response(prompt, st.session_state.current_character)
    
    if response_text and voice_id:
        with st.spinner(f"Generating {st.session_state.current_character}'s voice..."):
            response_audio = text_to_speech(
                response_text, 
                voice_id,
                stability=stability,
                similarity_boost=similarity_boost
            )
        
        current_history.append({
            "role": "assistant", 
            "content": response_text,
            "audio": response_audio.getvalue() if response_audio else None
        })
        
        with st.chat_message("assistant", avatar=avatar_mapping[st.session_state.current_character]):
            st.markdown(response_text)
            if response_audio:
                st.audio(response_audio, format="audio/mpeg")
    elif response_text:
        current_history.append({
            "role": "assistant", 
            "content": response_text,
        })
        
        with st.chat_message("assistant", avatar=avatar_mapping[st.session_state.current_character]):
            st.markdown(response_text)
            st.warning("Voice generation failed. Check your ElevenLabs API key.")
    else:
        st.error("Failed to generate character response")