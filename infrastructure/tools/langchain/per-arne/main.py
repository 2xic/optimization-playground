from whisper import Whisper
from reason import Reason
 
if __name__ == "__main__":
    """
    -> Per Arne greats the caller
    -> Caller says something (stream the audio in chunks)
    -> *Per Arne thinks*
    -> Per Arne uses a voice model to talk back :)
    -> Loop until people say goodbye.
    """
    call_info = Whisper().get_transcript("record.wav")
    agent = Reason()
    response = agent.predict(call_info)
    print(response)

    response2 = agent.predict(
        "ja, det skal være sendt en innbetaling. Kan du sjekke det ? "
    )
    print(response2)

