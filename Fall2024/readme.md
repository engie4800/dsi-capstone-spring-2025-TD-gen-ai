# Local setup instructions
### (1) Install Anaconda if not already installed (mainly for creating a separate environment, you can use plain python as well)
### (2) Save the secrets.json file in the root of the Fall2024 folder
### (3) Setup a new environment (python 3.12 is required as some packages don't support later versions)
conda create --name tdbank24 python=3.12
### (4) Activate new environment
conda activate tdbank24
### (5) Install the requirements for the new environment, we have to use pip because some packages are not available thru conda sources
pip install -r requirements.txt
### (6) Run streamlit locally (if you'r running on Windows prefix below command with start)
streamlit run chatbot.py


