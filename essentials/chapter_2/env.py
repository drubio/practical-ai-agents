from dotenv import load_dotenv
load_dotenv()

import os
api_key=os.getenv("ANTHROPIC_API_KEY")
print(api_key)
