# MediBot — Hospital FAQ Assistant

MediBot is a Streamlit-oriented conversational prototype for answering common hospital-service questions. The documented design combines a predefined FAQ dataset with a Google Gemini fallback for questions that do not match the FAQ content.

## Features described by the project

- FAQ content stored in `hospital_chatbot.csv`
- CSV-based question and answer data
- Gemini API integration for a generative fallback
- Streamlit user interface
- Python dependencies listed in `requirements.txt`

## Technology

- Python
- Streamlit
- pandas
- python-dotenv
- Google Generative AI SDK

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure the Gemini API key locally. Do not commit secrets.

   ```env
   GOOGLE_API_KEY=replace-with-your-api-key
   ```

4. Review the application entry point and run it with Streamlit once the source file is present in your local checkout.

## Data and safety considerations

- The included CSV is sample FAQ content, not a source of authoritative medical advice.
- Keep hospital-specific contact details, prices, and service information clearly labeled as sample data until verified by the relevant organization.
- Do not use this prototype to diagnose conditions, recommend treatment, or handle emergencies.
- Avoid uploading identifiable patient information to external model APIs.
- Treat model-generated answers as unverified and provide a clear route to a qualified human contact.

## Current documentation limitation

The repository includes a requirements file and FAQ dataset. The application source file was not located at the repository root during this review, so its runtime flow and Gemini prompt handling have not been independently verified.

## Suggested improvements

- Add the application source and a reproducible run guide.
- Add FAQ matching evaluation and fallback tests.
- Add input validation, rate limiting, and exception handling.
- Make the FAQ dataset configurable and remove duplicated sample entries.
- Add explicit disclaimers and human escalation for medical or emergency topics.
