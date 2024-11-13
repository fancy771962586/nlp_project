from openai import OpenAI
from configs import OPENAI_APIKEY,OPENAI_URL,OPENAI_MODEL_NAME


def chat_llm(sys_prompt, usr_prompt):
    client = OpenAI(base_url=OPENAI_URL, api_key=OPENAI_APIKEY)
    completion = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt}
        ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


if __name__ == '__main__':
    pass