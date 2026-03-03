# Agentic Chat bot
E-commerce chat-bot for assisting client with their purchase and FAQ
 
## Pre-requisite
- python 3.11 
- Ollama for serving llm models

## Installation
- Clone repo
```bash
git clone https://github.com/YaredAlex/agent-bot.git
```
- Create environment
```bash
python -m venv env
# activate environment
pip install -r requirements.txt
```
- Running Agent chatbot
```bash
# run mcp server first 
python .\src\mcp_server\server.py
# run agent 
python .\src\app\app.py
```
