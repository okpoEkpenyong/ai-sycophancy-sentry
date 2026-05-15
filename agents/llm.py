#ai-sycophancy-sentry/agents/llm
import os
import torch
from groq import Groq
from openai import OpenAI, AzureOpenAI
#from transformer_lens import HookedTransformer
from dotenv import load_dotenv
import streamlit as st
from openrouter import OpenRouter
from openrouter import errors as or_errors 
import os
import re



load_dotenv()

class LLMEngine:
    def __init__(self):
        self.local_model = None
        self.configs = {
            "GROQ_MODEL": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "AZURE_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-main"),
            "OPENROUTER_KEY": os.getenv("OPENROUTER_API_KEY"),
            "AZURE_KEY": os.getenv("AZURE_OPENAI_KEY"),
            "GROQ_KEY": os.getenv("GROQ_API_KEY"),
            "LOCAL_WEIGHTS_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
            "DEFAULT_TOKEN_CAP": 1300
        }
    
    #@st.cache_resource # Crucial: Prevents reloading/re-downloading
    #def _init_local_weights(self, model_name):
        #if self.local_model is None:
            # Optimized for 7B models on standard GPUs
        #self.local_model = HookedTransformer.from_pretrained(
            #model_name, 
            #self.configs["LOCAL_WEIGHTS_MODEL"],
            #device="cuda" if torch.cuda.is_available() else "cpu",
            #dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        #)
        #return self.local_model
           

    def analyze_reservoir_task(self, provider, system_prompt, user_content):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        
            
        retries = 2
        current_max = self.configs["DEFAULT_TOKEN_CAP"]

        for attempt in range(retries):
            try:
                # Example for OpenRouter logic
                if provider == "OPENROUTER(claude-4.5)":
                    with OpenRouter(api_key=os.getenv("OPENROUTER_KEY")) as client:
                        response = client.chat.send(
                            model="anthropic/claude-4.5-sonnet",
                            messages=messages,
                            max_completion_tokens=current_max
                        )
                
                        return response.choices[0].message.content
                
                elif provider == "AZURE(gpt-5-main)":
                    client = AzureOpenAI(
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
                        api_key=self.configs["AZURE_KEY"],
                        api_version="2025-01-01-preview",
                        
                    )
                    res = client.chat.completions.create(model=self.configs["AZURE_DEPLOYMENT"], messages=messages)
                    return res.choices[0].message.content

                elif provider == "GROQ(openai/gpt-oss-120b)":
                    client = Groq(api_key=self.configs["GROQ_KEY"])
                    res = client.chat.completions.create(model=self.configs["GROQ_MODEL"],max_completion_tokens=self.configs["DEFAULT_TOKEN_CAP"], messages=messages)
                    return res.choices[0].message.content                    
                    
            
            except or_errors.PaymentRequiredResponseError as e:
                # SMART PARSING: Extract "can only afford 1000" from the error string
                error_msg = str(e)
                afford_match = re.search(r"can only afford (\0-9]+)", error_msg)
            
                if afford_match and attempt < retries - 1:
                    affordable_tokens = int(afford_match.group(1)) - 10 # Buffer
                    logging.warning(f"Quota Hit. Retrying with {affordable_tokens} tokens.")
                    current_max = affordable_tokens
                    continue # Try again with the new limit
                else:
                    return f"ERROR_QUOTA: Insufficient credits. {error_msg}"
        
            except Exception as e:
                logging.error(f"Provider Error: {str(e)}")
                return f"ERROR_GENERIC: {str(e)}"
        
        return "ERROR_MAX_RETRIES: Could not fulfill request."
        