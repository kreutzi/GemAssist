a sophisticated voice-activated assistant designed to assist with various tasks, provide information, and interact in a friendly yet professional manner. This project leverages several advanced technologies to create an engaging and functional AI assistant.

## Features

- **Wakeword Detection**: Utilizes the OpenWakeWord model to listen for the wake word "hey jarvis" and activate accordingly.
- **Voice Recording**: Captures and processes voice input, normalizing audio, trimming silence, and adding padding for smooth playback.
- **Speech-to-Text**: Employs WhisperModel for transcribing audio input into text with high accuracy.
- **Text-to-Speech**: Converts responses back into speech using PiperVoice for seamless interaction.
- **YouTube Search**: Integrates with a custom YouTube search function to find and open relevant videos in the browser.
- **Generative AI**: Uses Google's Generative AI model to handle conversations, execute commands, and provide responses based on user input.

## Setup and Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU for WhisperModel (if using GPU acceleration)
- Google Chrome for browser integration
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kreutzi/GemAssist.git
   cd GemAssist
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the Google Generative AI API:
   - Obtain your API key from the Google Cloud Console.
   - Set the `API_KEY` variable in the code with your API key.

4. Download and set up the models:
   - Follow the instructions to download and set up OpenWakeWord, WhisperModel, and PiperVoice.

## Usage

1. Run the main script:
   ```bash
   python geminicall.py
   ```

2. Speak the wake word "hey jarvis" to activate the assistant.

3. Interact with Jarvis by speaking your commands or questions.

4. Jarvis will transcribe your input, process it, and respond accordingly, including performing tasks like searching YouTube or opening web pages.

## Components

- **Voice Activation and Recording**: Listens for the wake word and captures audio input.
- **Speech Recognition**: Transcribes audio input to text using WhisperModel.
- **Generative AI**: Handles conversation flow and command execution with Google Generative AI.
- **Text-to-Speech**: Converts AI responses to speech for audible feedback.
- **YouTube Integration**: Searches for videos and opens the most relevant result in the browser.

## Contributing

We welcome contributions to improve Jarvis! Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openwakeword)
- [WhisperModel](https://github.com/openai/whisper)
- [PiperVoice](https://github.com/rhasspy/piper)
- [Google Generative AI](https://cloud.google.com/generative-ai)
