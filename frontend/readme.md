# Local setup instructions
### (1) Install Anaconda if not already installed (mainly for creating a separate environment, you can use plain python as well)
### (2) Save the secrets.json file in the root of the Fall2024 folder
### (3) Setup a new environment (python 3.12 is required as some packages don't support later versions)
conda create --name tdbank24 python=3.12
### (4) Activate new environment
conda activate tdbank24
### (5) Install the requirements for the new environment, we have to use pip because some packages are not available thru conda sources
pip install -r requirements.txt

# Below are updated instructions for the new chatbot that supports openai, azure and ollama 

Note that we should still use the pinecone API key from the previous semester with openai as that is the only way to access the td-bank-docs (dimension: 1536) index that was created by the previous team.

The new documents loaded this semester are using a new index *td-bank-docs-new* (dimension: 768) This new index is not compatible yet with openai, you have to use ollama to access this index.

To make a new td-bank-docs index with new uploaded documents requires using openai and additional API charges. We can do this after we running some tests.


### (6) Install ollama for local development from here: https://ollama.com/

### (7) Pull required ollama models for local development

- ollama pull mistral
- ollama pull nomic-embed-text

### (8) Ensure that you have apprpriate secrets file named as follwows for each backend

Below is skeleton JSON for each file, assigned API keys are masked.

- secrets_openai.json

```json
{
    "openai_api_key": "ASSIGNED_API_KEY_HERE",
    "openai_api_endpoint": "https://api.openai.com/v1",
    "pinecone_api_key": "ASSIGNED_API_KEY_HERE",
    "langchain_api_key": "ASSIGNED_API_KEY_HERE",
    "cohere_api_key": "ASSIGNED_API_KEY_HERE"
}
```

- secrets_ollama.json

```json
{
    "openai_api_key": "ollama",
    "openai_api_endpoint": "http://localhost:11434/v1",
    "pinecone_api_key": "ASSIGNED_API_KEY_HERE",
    "langchain_api_key": "ASSIGNED_API_KEY_HERE",
    "cohere_api_key": "ASSIGNED_API_KEY_HERE"    
}
```

- secrets_azure.json (only create if you use Azure)

```json
{
    "openai_api_key": "ASSIGNED_API_KEY_HERE",
    "openai_api_endpoint": "ASSIGNED_API_ENDPOINT_HERE",
    "openai_api_version": "2024-08-01-preview",
    "pinecone_api_key": "ASSIGNED_API_KEY_HERE",
    "langchain_api_key": "ASSIGNED_API_KEY_HERE",
    "cohere_api_key": "ASSIGNED_API_KEY_HERE"    
}
```

### (9) Run streamlit locally

Streamlit does not support passing command line arguments to ivoked python scripts, so we have to pass through the OS environment. I created shell scripts for that below.

- For Windows: Use either start_streamlit_openai.bat or start_streamlit_ollama.bat
- For Linux: Use either start_streamlit_openai.sh or start_streamlit_ollama.sh
- For  Mac: Use either ./start_streamlit_openai.sh or ./start_streamlit_ollama.sh


