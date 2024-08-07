
import asyncio
import json
import logging
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from concurrent.futures import TimeoutError as ConnectionTimeoutError
import httpx
from pydantic import BaseModel
import requests
from retell import Retell
from AWS.migrate_to_s3 import move_files_to_user_folders
from api_requests.Transcriptparser import read_docx, setup_model
from custom_types import (
    ConfigResponse,
    ResponseRequiredRequest,
)
from llm import LlmClient  # or use .llm_with_func_calling
from fastapi import FastAPI
from fastapi.logger import logger

load_dotenv(override=True)
app = FastAPI()
retell = Retell(api_key=os.environ["RETELL_API_KEY"])


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
import boto3
from io import BytesIO

def download_file_from_s3(bucket_name, prefix, candidate_id):
    s3 = boto3.client('s3', region_name='us-east-1')
    file_key = f'{prefix}{candidate_id}/{candidate_id}.docx'
    try:
        s3_object = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = s3_object['Body'].read()
        return file_content
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        return None

def process_file_content(content, output_folder, job_description, company_background, call_details):
    file_id = "output"
    output_filename = f'{file_id}_parsed_transcript.json'
    output_filepath = os.path.join(output_folder, output_filename)

    chain = setup_model()
    response = chain.invoke({
        "job_description": job_description,
        "company_background": company_background,
        "content": content,
        "call_details": call_details
    })
    response_parsed = json.loads(response.json())

    with open(output_filepath, 'w') as f:
        json.dump(response_parsed, f, indent=4)
    print(response_parsed)

    print(f"Response saved to {output_filename}")
    return output_filepath

class DetailsRequestBody(BaseModel):
    candidate_id: int
    job_description: str
    company_details: str
    resume_analysis: str

@app.post("/details_from_db")
async def details_from_db(request: Request, body: DetailsRequestBody):
    # Construct payload for handle_webhook
    payload = {
        "event": "call_ended",  # or appropriate event type based on your logic
        "data": {
            "call_id": body.candidate_id
        }
    }

    # Pass the payload to handle_webhook
    try:
        # Assuming handle_webhook is modified to accept parameters directly
        response = await handle_webhook(Request(
            body=json.dumps(payload),
            headers={"x-retell-signature": "dummy_signature"}  # Provide a dummy or actual signature if needed
        ))
        return response
    except Exception as e:
        logging.error(f"Error processing details_from_db: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
# Your existing handle_webhook function
@app.post("/get_transcript")
async def handle_webhook(request: Request, x_retell_signature: str = Header(None)):
    try:
        post_data = await request.json()
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":")),
            api_key=os.environ["RETELL_API_KEY"],
            signature=x_retell_signature,
        )
        if not valid_signature:
            logging.error("Received Unauthorized: %s, %s", post_data["event"], post_data["data"]["call_id"])
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})

        event = post_data.get("event")
        data = post_data.get("data", {})
        call_id = data.get("call_id")

        if event == "call_started":
            logging.info("Call started event: %s", call_id)
        elif event == "call_ended":
            logging.info("Call ended event: %s", call_id)
            if not call_id:
                return JSONResponse(content={"error": "Call ID not found in the request data"}, status_code=400)
            
            token = os.getenv("RETELL_API_KEY")
            if not token:
                return JSONResponse(content={"error": "API token not found"}, status_code=500)
            
            try:
                call_details = get_call_details(call_id, token)  # Assuming get_call_details is synchronous, adjust if not
            except requests.RequestException as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)
            
            call_id = call_details.get('call_id')
            transcript = call_details.get('transcript')
            call_analysis = call_details.get('call_analysis')
            call_summary = call_analysis.get('call_summary') if call_analysis else None

            logging.info("Call ID: %s", call_id)
            logging.info("Transcript: %s", transcript)
            logging.info("Call Analysis: %s", call_analysis)
            logging.info("Call Summary: %s", call_summary)
            
            store_call_details_url = "http://127.0.0.1:8500/store_call_details/"
            store_call_details_payload = {
                "candidate": 1,
                "pdf_url": "http://example.com/file.pdf",
                "call_summary": call_summary,
                "transcript": transcript,
                "resume_analysis": "call_details",
            }
            headers = {
                'Content-Type': 'application/json',
            }
            async with httpx.AsyncClient() as client:
                store_call_details_response = await client.post(store_call_details_url, json=store_call_details_payload, headers=headers)

            if store_call_details_response.status_code != 201:
                return JSONResponse(content={"message": "Failed to store call details", "status": store_call_details_response.status_code, "error": store_call_details_response.text})
            
            bucket_name = 'sample-candidates-pluto-dev'
            prefix = 'user_data_1/'
            local_folder = '/transcripts'
            
            file_content = move_files_to_user_folders(local_folder, bucket_name, prefix)
            if not file_content:
                return JSONResponse(content={"error": "Failed to download file from S3"}, status_code=500)

            job_description = "Senior Data Scientist at Moderna. This role involves developing and overseeing Immuno-Assay development, essential for Moderna's research and vaccine production efforts."
            company_background = "Moderna is a biotechnology company pioneering messenger RNA (mRNA) therapeutics and vaccines. We aim to transform how medicines are created and delivered, focusing on preventing and fighting diseases."
            transcripts_folder = 'transcripts'

            content = read_docx(BytesIO(file_content))  # Assuming read_docx is a function to read docx content
            output_filepath = process_file_content(content, transcripts_folder, job_description, company_background, call_details)
            
            move_files_to_user_folders(transcripts_folder, bucket_name, prefix)
        
        elif event == "call_analyzed":
            logging.info("Call analyzed event: %s", call_id)
        else:
            logging.warning("Unknown event: %s", event)
        
        return JSONResponse(status_code=204, content={"message": "Event processed successfully"})
    except Exception as err:
        logging.error(f"Error in webhook: {err}")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})