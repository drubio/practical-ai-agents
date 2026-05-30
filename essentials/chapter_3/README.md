# Direct LLM API access: Anthropic, OpenAI, Google, xAI, DeepSeek

This chapter demonstrates how to access the major LLM providers—Anthropic, OpenAI, Google, xAI and DeepSeek—using their in-house APIs and minimal working scripts in both Python and JavaScript.

## Project Contents
Each subfolder reflects an LLM provider and contains:

- A `nursery.py` script (Python)
- A `nursery.js` script (JavaScript)
- The necessary `package.json` / `requirements.txt` for dependencies

The scripts generate and test completions using the nursery rhyme:  
`"Twinkle, Twinkle, Little"`, to verify LLM functionality and output quality.

*NOTE*: An `.env` file with the corresponding LLM API keys must also be generated directly in this folder, like it was done in the last chapter
---


## Environment Setup

For each provider (`anthropic`, `openai`, `google`, `xai`, `deepseek`):

### ❯ Example (Python)

### Step 1. Navigate into the provider folder:
   ```bash
   cd anthropic  # or openai, google, xai, deepseek
````

### Step 2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

NOTE: This chapter does **not** use the repository-level shared libraries in `shared/`

### Step 3. Create an `.env` file with the correct LLM APIs keys like it was done in the last chapter:

   ```env
   ANTHROPIC_API_KEY="sk-ant-api03-xxxxxx"
   OPENAI_API_KEY="sk-proj-xxxx-xxxx-xxxx"
   GOOGLE_API_KEY="AIzaSyBAKuqxxxxxxxxxxxx"
   XAI_API_KEY="xai-UXxxxxxxx"
   DEEPSEEK_API_KEY="sk-e0fxxxxxxx"
   ```

### Step 4. Run the script:

   ```bash
   python nursery.py
   ```

---

### > Example (JavaScript)

### Step 1. Navigate into the provider folder:

   ```bash
   cd anthropic  # or openai, google, xai, deepseek
   ```

### Step 2. Install dependencies:

   ```bash
   npm install
   ```

NOTE: This chapter does **not** use the repository-level shared libraries in `shared/`

### Step 3. Create an `.env` file with the correct LLM API keys like it was done in the last chapter (or described above).

### Step 4. Run the script:

   ```bash
   node nursery.js
   ```

---

## Summary

Each script:

* Loads its provider LLM  API key from `.env`
* Sends the prompt `"Twinkle, Twinkle, Little"` to the model
* Outputs the model's prediction
* Uses the minimum code requirements for Python and JavaScript.

This setup for each provider also gives a clear baseline for model behavior across providers before using higher-level frameworks like LangChain or LlamaIndex.
