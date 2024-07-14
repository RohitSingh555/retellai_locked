import json
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from concurrent.futures import TimeoutError as ConnectionTimeoutError
import requests
from retell import Retell
from AWS.migrate_to_s3 import move_files_to_user_folders
from api_requests.Transcriptparser import read_docx, setup_model
from custom_types import (
    ConfigResponse,
    ResponseRequiredRequest,
)
from llm import LlmClient  # or use .llm_with_func_calling

load_dotenv(override=True)
app = FastAPI()
retell = Retell(api_key=os.environ["RETELL_API_KEY"])

# twilio_client.create_phone_number(213, "68978b1c2935ff9c7d7107e61524d0bb")
# twilio_client.delete_phone_number("+12133548310")
# twilio_client.register_inbound_agent("+13392016322", "68978b1c2935ff9c7d7107e61524d0bb")
# twilio_client.create_phone_call("+13392016322", "+14157122917", "68978b1c2935ff9c7d7107e61524d0bb")


# Handle webhook from Retell server. This is used to receive events from Retell server.
# Including call_started, call_ended, call_analyzed
@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        post_data = await request.json()
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":")),
            api_key=str(os.environ["RETELL_API_KEY"]),
            signature=str(request.headers.get("X-Retell-Signature")),
        )
        if not valid_signature:
            print(
                "Received Unauthorized",
                post_data["event"],
                post_data["data"]["call_id"],
            )
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})
        if post_data["event"] == "call_started":
            print("Call started event", post_data["data"]["call_id"])
        elif post_data["event"] == "call_ended":
            print("Call ended event", post_data["data"]["call_id"])
        elif post_data["event"] == "call_analyzed":
            print("Call analyzed event", post_data["data"]["call_id"])
        else:
            print("Unknown event", post_data["event"])
        return JSONResponse(status_code=200, content={"received": True})
    except Exception as err:
        print(f"Error in webhook: {err}")
        return JSONResponse(
            status_code=500, content={"message": "Internal Server Error"}
        )



# Start a websocket server to exchange text input and output with Retell server. Retell server
# will send over transcriptions and other information. This server here will be responsible for
# generating responses with LLM and send back to Retell server.
@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    try:
        await websocket.accept()
        llm_client = LlmClient()

        # Send optional config to Retell server
        config = ConfigResponse(
            response_type="config",
            config={
                "auto_reconnect": True,
                "call_details": True,
            },
            response_id=1,
        )
        await websocket.send_json(config.__dict__)

        # Send first message to signal ready of server
        response_id = 0
        first_event = llm_client.draft_begin_message()
        await websocket.send_json(first_event.__dict__)

        async def handle_message(request_json):
            nonlocal response_id

            # There are 5 types of interaction_type: call_details, pingpong, update_only, response_required, and reminder_required.
            # Not all of them need to be handled, only response_required and reminder_required.
            if request_json["interaction_type"] == "call_details":
                print(json.dumps(request_json, indent=2))
                return
            if request_json["interaction_type"] == "ping_pong":
                await websocket.send_json(
                    {
                        "response_type": "ping_pong",
                        "timestamp": request_json["timestamp"],
                    }
                )
                return
            if request_json["interaction_type"] == "update_only":
                return
            if (
                request_json["interaction_type"] == "response_required"
                or request_json["interaction_type"] == "reminder_required"
            ):
                response_id = request_json["response_id"]
                request = ResponseRequiredRequest(
                    interaction_type=request_json["interaction_type"],
                    response_id=response_id,
                    transcript=request_json["transcript"],
                )
                print(
                    f"""Received interaction_type={request_json['interaction_type']}, response_id={response_id}, last_transcript={request_json['transcript'][-1]['content']}"""
                )

                async for event in llm_client.draft_response(request):
                    await websocket.send_json(event.__dict__)
                    if request.response_id < response_id:
                        break  # new response needed, abandon this one

        async for data in websocket.iter_json():
            asyncio.create_task(handle_message(data))

    except WebSocketDisconnect:
        print(f"LLM WebSocket disconnected for {call_id}")
    except ConnectionTimeoutError as e:
        print("Connection timeout error for {call_id}")
    except Exception as e:
        print(f"Error in LLM WebSocket: {e} for {call_id}")
        await websocket.close(1011, "Server error")
    finally:
        print(f"LLM WebSocket connection closed for {call_id}")
        
from api_requests.api_requests import get_call_details


def process_files_in_folder(folder_path, output_folder, job_description, company_background):
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_id = filename.split('_')[0]
            output_filename = f'{file_id}_parsed_transcript.json'
            output_filepath = os.path.join(output_folder, output_filename)
            
            if os.path.exists(output_filepath):
                print(f"Skipping {filename} as {output_filename} already exists.")
                continue
            
            file_path = os.path.join(folder_path, filename)
            content = read_docx(file_path)

            chain = setup_model()
            response = chain.invoke({"job_description": job_description, "company_background": company_background, "content": content})
            response_parsed = json.loads(response.json())

            with open(output_filepath, 'w') as f:
                json.dump(response_parsed, f, indent=4)
            print(response_parsed)

            print(f"Response saved to {output_filename}")

@app.post("/get_transcript")
async def get_transcript(request: Request):
    data = await request.json()
    print(f"Received data: {data}")

    event = data.get("event")
    if event != "call_ended":
        return JSONResponse(content={"error": "Unsupported event type"}, status_code=400)

    call_id = data.get("call", {}).get("call_id")
    print(f"Call ID: {call_id}")
    if not call_id:
        return JSONResponse(content={"error": "Call ID not found in the request data"}, status_code=400)
    
    token = os.getenv("RETELL_API_KEY")
    if not token:
        return JSONResponse(content={"error": "API token not found"}, status_code=500)
    
    try:
        call_details = get_call_details(call_id, token)
    except requests.RequestException as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    print("\n\n\n")
    print(call_details)

    job_description = "Senior Data Scientist at Moderna. This role involves developing and overseeing Immuno-Assay development, essential for Moderna's research and vaccine production efforts."
    company_background = "Moderna is a biotechnology company pioneering messenger RNA (mRNA) therapeutics and vaccines. We aim to transform how medicines are created and delivered, focusing on preventing and fighting diseases."
    transcripts_folder = 'transcripts'
    output_folder = transcripts_folder 

    process_files_in_folder(output_folder, output_folder, job_description, company_background)
    
    bucket_name = 'sample-candidates-pluto-dev'
    prefix = 'user_data/'
    local_folder = 'transcripts/'
    
    move_files_to_user_folders(local_folder, bucket_name, prefix)

    return JSONResponse(content={"message": "Data received and processed"})