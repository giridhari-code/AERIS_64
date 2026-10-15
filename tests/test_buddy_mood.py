from sentinel.buddy.mood import mood_from_text

def test_question_think():
    assert mood_from_text("How does memory work?", role="user") == "think"

def test_hello_wave():
    assert mood_from_text("Namaste", role="user") == "wave"

def test_assistant_talk():
    assert mood_from_text("Here is the answer.", role="assistant") == "talk"
