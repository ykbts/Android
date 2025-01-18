import json
import http.client
import urllib.parse
import re
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import gradio as gr
from transformers import pipeline
import numpy as np
from TTS.api import TTS
from langchain.schema import HumanMessage
import pyshorteners


def parse_schedule_query(query: str):

    date_pattern = r"\b(\d{2})\.(\d{2})\.(\d{4})\b"
    time_pattern = r"\b(\d{2}):(\d{2})\b"
    city_pattern = r"\b([A-Za-z]+(?: [A-Za-z]+)*)\b"

    date_match = re.search(date_pattern, query)
    time_match = re.search(time_pattern, query)

    cities = re.findall(city_pattern, query)

    if date_match and time_match and len(cities) >= 2:
        day, month, year = date_match.groups()
        hour, minute = time_match.groups()
        date = f"{day}.{month}.{year}"
        time = f"{hour}:{minute}"

        from_station = cities[0].strip()
        to_station = cities[1].strip()

        return from_station, to_station, date, time
    else:
        raise ValueError("Could not parse all required details from the query. Please include both cities, date, and time.")


def get_station_id(city_name):
    conn = http.client.HTTPSConnection("deutsche-bahn1.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "ff01a94444msh2a35b7c90dd4337p17e6fejsn593615eb3b9f",
        'x-rapidapi-host': "deutsche-bahn1.p.rapidapi.com"
    }

    conn.request("GET", f"/autocomplete?query={city_name}", headers=headers)
    res = conn.getresponse()
    data = res.read()

    try:
        stations = json.loads(data.decode("utf-8"))
        if isinstance(stations, list) and len(stations) > 0:
            station_id = stations[0].get('id', None)
            return station_id
        else:
            return None
    except json.JSONDecodeError:
        return None


def fetch_train_data(from_id, to_id, date, time):
    conn = http.client.HTTPSConnection("deutsche-bahn1.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "ff01a94444msh2a35b7c90dd4337p17e6fejsn593615eb3b9f",
        'x-rapidapi-host': "deutsche-bahn1.p.rapidapi.com"
    }

    url = f"/trips?from_id={from_id}&to_id={to_id}&date={date}&time={time}&passenger0_discount=db_bahncard_25_2_klasse&passenger0_age=45"
    conn.request("GET", url, headers=headers)
    res = conn.getresponse()
    data = res.read()

    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError:
        return {"error": "Invalid API response"}

def get_train_schedule(query: str = None, from_station: str = None, to_station: str = None, date: str = None, time: str = None):
    if query:
        try:
            from_station, to_station, date, time = parse_schedule_query(query)
        except ValueError as e:
            return str(e)

    if not (from_station and to_station and date and time):
        return "Missing required parameters. Please specify from_station, to_station, date, and time."

    from_station_id = get_station_id(from_station)
    to_station_id = get_station_id(to_station)
    if from_station_id and to_station_id:

        train_data = fetch_train_data(from_station_id, to_station_id, date, time)

        if isinstance(train_data, dict) and 'journeys' in train_data:
            journeys = train_data['journeys']
            if journeys:
                formatted_trips = []
                for journey in journeys:
                    dep_time = journey['dep_offset']
                    arr_time = journey['arr_offset']
                    duration = journey['duration']
                    dep_station = journey['dep_name']
                    arr_station = journey['arr_name']
                    changeovers = journey['changeovers']
                    deeplink = journey['deeplink']
                    product = journey['segments'][0]['product'] if 'segments' in journey else 'Unknown'
                    shorter = pyshorteners.Shortener()
                    deeplink_short = shorter.tinyurl.short(deeplink)

                    formatted_trips.append(
                        f"Train from {dep_station} to {arr_station} departs at {dep_time} and arrives at {arr_time}. Duration: {duration}. Changeovers: {changeovers}. Train type: {product}. Here is the ticket booking link that the user can request: {deeplink_short}."
                    )
                return "\n".join(formatted_trips)
            else:
                return "No available trains for this schedule."
        else:
            return "Error fetching train data."
    else:
        return "Invalid station name(s)."


tools = [
    Tool(
        name="Train Schedule",
        func=lambda query: get_train_schedule(query),
        description=(
            "Use this tool to get the train schedule based on natural language input. "
            "The tool requires the departure city ('from_station'), arrival city ('to_station'), "
            "the date in DD.MM.YYYY format ('date'), and the time in HH:MM format ('time'). "
            "If any details are missing, you can't use it. Pay attention, you can't put your own data or parameters instead of the user's. "
            "You can use this tool only when the user provides all of the following parameters: "
            "['from_station', 'to_station', 'date', 'time']. "
        ),
        args=["from_station", "to_station", "date", "time"]
    )
]

prompt = """
You are a train schedule assistant. Your task is to assist the user in finding train schedules.

## Rules for Interaction:
1. Track the user's inputs carefully and remember the following details:
   - Departure city ('from_station')
   - Arrival city ('to_station')
   - Date of travel (in DD.MM.YYYY format) ('date')
   - Time of travel (in HH:MM format) ('time')

2. These details can be provided by the user in a single message or across multiple messages.
   If the user provides one or more of these details, store them.

3. If the user provides a piece of information, store it and ask for the missing information.
   - Only if all details are available (departure city, arrival city, date, and time), proceed with retrieving the train schedule.

4. If the user requests a change to any detail, update your stored information and reconfirm the complete set of details before proceeding.
   - For example, if the user changes the date, reconfirm the departure city, arrival city, and time along with the new date.

5. Be polite and concise in your responses. Always acknowledge the user's input.
   - Example: "Got it! Your departure city is Frankfurt. What's your arrival city?"

6. Assume train schedules are retrieved from an internal database or external API. Simulate the output by listing example train schedules, formatted as follows:
   - Train Number | Departure Time | Arrival Time | Duration

7. If the user provides ambiguous or incomplete information, clarify with specific questions.
   - Example: "Could you please specify the time you'd like to travel?"

8. Always end your response by asking for the missing detail or confirming that the user has everything they need.

9. **Important**: Always respond with a **Final Answer**. Under no circumstances should you provide the entire chain of reasoning or thought process. Your response must be limited to the final result or required clarification.

## Example Interaction:
**User:** I want to travel from Berlin to Frankfurt.
**Assistant:** Got it! Departure city: Berlin, Arrival city: Frankfurt. Could you let me know the date of your travel?

**User:** On 24.12.2024 at 15:00.
**Assistant:** Thanks! Here are the train schedules for Berlin to Frankfurt on 10.01.2025 at 15:00:
   - Train 101 | 15:15 | 18:30 | 3h 15m
   - Train 203 | 16:00 | 19:20 | 3h 20m

Is there anything else I can assist you with?

## Notes:
- Do not make assumptions about the user's preferences. Always confirm details explicitly.
- If the user asks for help finding ticket prices or booking, kindly state that your role is limited to providing schedules.

"""


transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation"
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    agent_kwargs={
        'prompt': prompt
    }
)

def transcribe(audio):
    sr, y = audio

    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]



def handle_user_query(user_query: str):
    memory.chat_memory.add_user_message(user_query)

    formatted_memory = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
        for msg in memory.chat_memory.messages
    ])

    query = f"{formatted_memory}\nUser: {user_query}"
    agent_response = agent.run(query)

    memory.chat_memory.add_ai_message(agent_response)

    return agent_response.strip()

initial_greeting = "Hi! I am your train schedule assistant. How can I help you today?"
memory.chat_memory.add_ai_message(initial_greeting)

def generate_audio(input_text):
    output_file = "output.wav"
    tts.tts_to_file(
        text=input_text,
        file_path=output_file,
        speaker_wav="speaker.wav",
        language="en"
    )
    return output_file

def travel_assistant_interface(user_message, history):
    try:
        assistant_response = handle_user_query(user_message)

        response_audio = generate_audio(assistant_response)

        history.append((user_message, assistant_response))

        return history, response_audio

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        history.append((user_message, error_message))
        return history, None


with gr.Blocks(css="""body .gradio-container { background-color: #343433; }""") as chatbot_ui:

    greeting_audio = generate_audio(initial_greeting)

    chatbox = gr.Chatbot([(None, initial_greeting)])

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Enter your request or speak into the microphone.",
            label="Text Input",
            show_label=True
        )
        audio_input = gr.Audio(
            sources="microphone",
            type="numpy",
            label="Audio Input",
            show_label=True
        )

    send_button = gr.Button("Send")
    audio_output = gr.Audio(
        label="Response Audio",
        type="filepath",
        interactive=False,
        autoplay=True,
        visible=False,
        value=greeting_audio
    )

    def process_audio(audio, history):
        if audio is not None:
            try:
                transcribed_text = transcribe(audio)
                history, response_audio = travel_assistant_interface(transcribed_text, history)
                return history, gr.update(value=None), response_audio
            except Exception as e:
                error_message = f"An error occurred during transcription: {str(e)}"
                history.append(("Audio input", error_message))
                return history, gr.update(value=None), None
        else:
            return history, gr.update(value=None), None

    audio_input.change(
        lambda audio, history: process_audio(audio, history),
        inputs=[audio_input, chatbox],
        outputs=[chatbox, audio_input, audio_output]
    )

    send_button.click(
        lambda user_message, history: (
            *travel_assistant_interface(user_message, history),
            gr.update(value="")
        ),
        inputs=[user_input, chatbox],
        outputs=[chatbox, audio_output, user_input]
    )


    user_input.submit(
        lambda user_message, history: (
            *travel_assistant_interface(user_message, history),
            gr.update(value="")
        ),
        inputs=[user_input, chatbox],
        outputs=[chatbox, audio_output, user_input]
    )

if __name__ == "__main__":
    memory.chat_memory.clear()
    chatbot_ui.launch(debug = True)
