import aiohttp  
import asyncio 
import nest_asyncio
import time

from tqdm import tqdm

from utils.model_configs import get_model_configs

# gemini_api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY", None)
# anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", None)
# perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", None)

class AsyncLLMPipeline:
    def __init__(self, model:str=None):
        self.model = model

    async def agenerate(self,
                      user_prompt,
                      max_retries:int=5):
        raise NotImplementedError

    async def model_response(self,
                             user_prompts):
        
        if not isinstance(user_prompts,list):
            user_prompts = [user_prompts]

        tasks = [asyncio.create_task(self.agenerate(user_prompt))
                 for user_prompt in user_prompts]
            
        responses = await asyncio.gather(*tasks)
        timeout_or_incorrect_resp = 0 #count for how many requests timed out or have incorrect responses
        
        decoded_responses = []
        for resp in responses:

            try:
                if self.model.startswith("gemini"):
                    decoded_response = resp["candidates"][0]["content"]["parts"][0]["text"]

                elif self.model.startswith("claude"):
                    decoded_response = resp["content"][0]["text"]

                elif self.model.startswith("perplexity"):
                    decoded_response = resp["text"]

                elif self.model.startswith("deepseek-r1") or "ollama" in self.model:
                    # Ollama with OpenAI-compatible API returns OpenAI format
                    decoded_response = resp['choices'][0]['message']['content']

                else:
                    decoded_response = resp['choices'][0]['message']['content']
            except (TypeError, KeyError) as e:
                print(f"Error decoding response: {e}")
                print(f"Response: {resp}")
                timeout_or_incorrect_resp += 1
                decoded_response = 'indeterminate'
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(f"Response: {resp}")
                decoded_response = 'indeterminate'
            finally:
                decoded_responses.append(decoded_response)

        print(f"{timeout_or_incorrect_resp} requests out of {len(tasks)} requests either timed out or returned non-parseable outputs ...")
            
        return decoded_responses

    def batch_generate(self,
                      user_prompts):
        
        batched_prompts = [user_prompts[idx : idx+50]
                           for idx in range(0, len(user_prompts), 50)]
        
        outputs = []
        for batch in tqdm(batched_prompts):
            batch_output = asyncio.run(self.model_response(batch))
            outputs.extend(batch_output)

            time.sleep(0.05) #sleep for a second after each batch is processed!

        return outputs

nest_asyncio.apply()

class LLM(AsyncLLMPipeline):

    def __init__(self,
                 system_prompt:str,
                 few_shot_examples:list=[],
                 model:str="gemini-2.0-flash",
                 max_timeout_per_request:int=15):

        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples
        self.model = model

        self.max_timeout_per_request = max_timeout_per_request

        # Get model-specific configurations
        self.model_configs = get_model_configs(model)

        print(f'Using model : {self.model}')

        super().__init__(model=self.model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        print('Exit called ... cleaning up')
        print('Cleanup complete!\n')

        return True

    async def agenerate(self,
                      user_prompt,
                      max_retries:int=2):

        retries = 0
        backoff_factor = 2
        min_sleep_time = 3

        messages = []

        system_prompt = [{"role" : "system", "content" : self.system_prompt}]
        messages.extend(system_prompt)

        if self.few_shot_examples != []:
            examples = [[{"role" : "user", "content" : examples[0]},{"role" : "assistant", "content" : examples[1]}]
                        for examples in self.few_shot_examples]
            
            examples = [arr for sublist in examples for arr in sublist]
            messages.extend(examples)

        user_prompt = [{"role" : "user", "content" : user_prompt}]
        messages.extend(user_prompt)

        while retries < max_retries:
            try:
                # Use the model-specific message formatter
                request_data = self.model_configs.message_formatter(messages)

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=self.model_configs.base_url,
                        headers=self.model_configs.headers,
                        json=request_data,
                        timeout=self.max_timeout_per_request
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"API request failed with status {response.status}: {error_text}")
            
            except asyncio.TimeoutError as timeout_err:
                print("\ntimeout err : ",timeout_err)
                print('request sent : ',messages)
                return 'indeterminate'
                    
            except Exception as e:
                print('Exception: {}'.format(e))
                sleep_time = min_sleep_time * (backoff_factor ** retries)
                print(f"Rate limit hit. Retrying in {sleep_time} seconds.")
                await asyncio.sleep(sleep_time) 
                retries += 1
        
        return 'indeterminate'