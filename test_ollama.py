import ollama

try:
    response = ollama.chat(model='llama3.1:8b', messages=[
        {'role': 'user', 'content': 'Say hello in one sentence.'}
    ])
    print("SUCCESS:")
    print(response['message']['content'])
except Exception as e:
    print("ERROR OCCURRED:")
    print(e)