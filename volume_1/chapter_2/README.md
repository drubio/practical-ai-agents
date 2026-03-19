# Environment Variable Setup with `.env` files

This chapter shows how to load API keys and other environment variables using `.env` files, in both Python and JavaScript.

## Project Contents

```
env.js                # JavaScript example using dotenv
env.py                # Python example using python-dotenv
rename_to_.env.txt    # Sample .env file—rename before use
requirements.txt      # Python dependency list
package.json          # JavaScript dependencies with ES module enabled
````

---

## Environment Setup

### Step 1 — Rename the .env File

Rename the provided sample file so it's recognized as a real environment config file:

```bash
mv rename_to_.env.txt .env
````

Edit the `.env` key values to reflect your own LLM API keys

Example `.env` file:

```env
OPENAI_API_KEY="sk-proj-xxxx-xxxx-xxxx"
GOOGLE_API_KEY="AIzaSyBAKuqxxxxxxxxxxxx"
ANTHROPIC_API_KEY="sk-ant-api03-xxxxxx"
XAI_API_KEY="xai-UXxxxxxxx"
```

---

### Step 2 — Run the Examples

### ❯ Example (Python)

**Install dependencies:**

```bash
pip install -r requirements.txt
```

NOTE: This chapter does **not** use the repository-level shared libraries in `shared/`


**Run the script:**

```bash
python env.py
```

This script loads `.env`, fetches the `ANTHROPIC_API_KEY`, and prints it:

```python
from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("ANTHROPIC_API_KEY")
print(api_key)
```

---

### ❯ Example (JavaScript)

**Install dependencies:**

```bash
npm install
```

NOTE: This chapter does **not** use the repository-level shared libraries in `shared/`

**Run the script:**

```bash
node env.js
```

This script loads `.env`, fetches the `OPENAI_API_KEY`, and prints it:

```js
import dotenv from 'dotenv';
dotenv.config();

const apiKey = process.env.OPENAI_API_KEY;
console.log(apiKey);
```
---

## Summary

Both the Python and JavaScript examples use standard libraries to safely load environment variables from a local `.env` file—a best practice for working with API keys.
