import random

# Define some possible user inputs and corresponding bot responses
responses = {
    "hello": ["Hi there!", "Hello!", "Hi!"],
    "how are you?": ["I'm doing well, thanks for asking.", "I'm fine, how about you?", "I'm good, thanks!"],
    "what's your name?": ["My name is Bot.", "I'm Bot, nice to meet you!", "You can call me Bot."],
    "bye": ["Goodbye!", "See you later!", "Bye!"]
}

# Define a function to get a response from the bot
def get_response(user_input):
    # Convert the user input to lowercase and remove any punctuation
    user_input = user_input.lower()
    
    # Check if the user input matches any of the predefined responses
    for key in responses.keys():
        if user_input == key:
            return random.choice(responses[key])
    
    # If no match was found, return a default response
    return "I'm sorry, I don't understand what you're saying."

# Define the main loop of the chatbot program
while True:
    # Get user input and exit if user enters "bye"
    user_input = input("You: ")
    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break
    
    # Get a response from the bot and print it
    bot_response = get_response(user_input)
    print("Bot:", bot_response)
