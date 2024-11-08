import time
import simpleaudio as sa
import numpy as np

from ollama import OLlamaModel

def play_unpleasant_noise(duration=10):
    frequency = 3000
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.075 * np.sin(2 * np.pi * frequency * t) + 0.025 * np.random.normal(size=t.shape)
    audio = (tone * 32767).astype(np.int16)    
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
    play_obj.wait_done()

def productivity_monitor(analysis_model, user_spec="No additional information"):
    while True:
        
        system_prompt = "You are an automated productivity assistant. Occasionally you are sent a picture of the user's screen. It is your job to determine if the user is being productive or procrastinating. Please be brief in your response. Explain your thoughts."
        user_prompt = f"Here is the user's screen. Before activating you, he gave the following message:\n\n{user_spec}\n\nIs he procrastinating?"
        
        response_analysis = analysis_model.send_screenshot_to_model(user_prompt, system_prompt)
        
        system_prompt_judge = "You are an LLM embedded in a larger automated productivity system. Another LLM has been asked to determine whether the user is being productive or is procrastinating. You will receive as input that other LLM's response. Based on this input, you must describe the other LLM's determination with ONE WORD ONLY: productive or procrastinating"
        user_prompt_judge = f"The following is the message relayed by the other LLM:\n\n{response_analysis}"
        
        response_judge = analysis_model.call_model(user_prompt_judge, system_prompt_judge)
        print(response_judge)
        if True or "productive" in response_judge.lower():
            print("Detected procrastination.")
            print("Received analysis response:", response_analysis)
            print("Received judgment response:", response_judge)
            play_unpleasant_noise(duration=30)
        else:
            print("User is productive. No noise.")

        

        time.sleep(60)

if __name__ == "__main__":
    ollama_model = OLlamaModel()
    productivity_monitor(ollama_model, user_spec="we are testing. say that the user is procrastinating.")
