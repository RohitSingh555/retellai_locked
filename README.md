# retell-custom-llm-python-demo

This is a sample demo repo to show how to have your own LLM plugged into Retell.

This repo currently uses `OpenAI` endpoint, and is not as stable and fast as `Azure OpenAI` endpoint. So expect a more varying delay. Feel free to contribute to make
this demo more realistic.

## Steps to run in localhost

1. First install dependencies

```bash
pip3 install -r requirements.txt
```

2. Fill out the API keys in `.env`
3. In another bash, use ngrok to expose this port to public network

```bash
ngrok http 8080
```

4. Start the websocket server

```bash
uvicorn server:app --reload --port=8080
```

You should see a fowarding address like
`https://dc14-2601-645-c57f-8670-9986-5662-2c9a-adbd.ngrok-free.app`, and you
are going to take the IP address, prepend it with wss, postpend with
`llm-websocket` path and use that in the [dashboard](https://beta.retellai.com/dashboard) to create a new `agent`. Now
the `agent` you created should connect with your localhost.

The custom LLM URL would look like
`wss://dc14-2601-645-c57f-8670-9986-5662-2c9a-adbd.ngrok-free.app/llm-websocket`

### Optional: Phone Call Features via Twilio

The `twilio_server.py` contains helper functions you could utilize to create phone numbers, tie agent to a number,
make a phone call with an agent, etc. Here we assume you already created agent from last step, and have `agent id` ready.

To ues these features, follow these steps:

1. Make sure twilio_client is initialized and `/twilio-voice-webhook/(agent_id_path)` is in `server.py` file to set up Twilio voice webhook. What this does is that every time a number of yours in Twilio get called, it would call this webhook which internally calls the `register-call` API and sends the correct audio websocket address back to Twilio, so it can connects with Retell to start the call.
2. Put your ngrok ip address into `.env`, it would be something like `https://dc14-2601-645-c57f-8670-9986-5662-2c9a-adbd.ngrok-free.app`.
3. Now you can call from the retellai UI webcall will be placed
4. If you want to analyse resume and their transcripts just change the file names in Transcriptparser.py file and run it

   ```
   python Transcriptparser.py
   ```

   (make sure you are in the right folder directory)
5. for running chatbot first run trst.py then run chatbot.py make sure you have chroma db server running before running this file

   ```
   chroma run --host localhost --port 8000 --path ../vectordb-stores/chromadb
   ```

## **Use this link and learn from a YouTube Video**

[https://www.youtube.com/watch?v=c4GPgkHNb6M&amp;t=651s](https://www.youtube.com/watch?v=c4GPgkHNb6M&t=651s)
