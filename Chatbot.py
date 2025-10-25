from transformers import pipeline
from collections import deque
import re
import random

class Chatbot:
    def __init__(self):
        print("🔄 Loading model... (distilgpt2 - PyTorch only)")
        self.generator = pipeline("text-generation", model="distilgpt2", framework="pt")
        print("✅ Model loaded successfully!\n")

        self.memory = deque(maxlen=3)
        self.last_topic = None
        print("🤖 Chatbot ready! Type '/exit' to quit.\n")

    def small_talk(self, user_input):
        """Handles greetings and simple conversations."""
        text = user_input.lower()

        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        thanks = ["thank you", "thanks", "appreciate it"]
        how_are_you = ["how are you", "how r u", "how are u"]
        bye = ["bye", "goodbye", "see you", "take care"]

        if any(word in text for word in greetings):
            return random.choice([
                "Hello there! 😊",
                "Hey! How are you doing today?",
                "Hi! Nice to see you again.",
                "Good to see you! What’s up?"
            ])

        elif any(word in text for word in how_are_you):
            return random.choice([
                "I'm just a bunch of code, but I'm feeling great today! 😄",
                "Doing awesome! How about you?",
                "I'm good, thanks for asking. How are you?"
            ])

        elif any(word in text for word in thanks):
            return random.choice([
                "You're welcome! 😊",
                "Glad I could help!",
                "Anytime! 😄"
            ])

        elif any(word in text for word in bye):
            return random.choice([
                "Goodbye! Have a great day! 👋",
                "See you later!",
                "Bye! Take care!"
            ])

        return None

    def find_fact(self, user_input):
        """Handles factual questions (e.g., capitals)."""
        facts = {
            "france": "The capital of France is Paris.",
            "italy": "The capital of Italy is Rome.",
            "germany": "The capital of Germany is Berlin.",
            "india": "The capital of India is New Delhi.",
            "japan": "The capital of Japan is Tokyo.",
            "usa": "The capital of the USA is Washington, D.C.",
            "china": "The capital of China is Beijing.",
            "russia": "The capital of Russia is Moscow.",
        }

        match = re.search(r"(?:about|of)\s+([A-Za-z]+)", user_input.lower())
        if match:
            country = match.group(1)
            if country in facts:
                self.last_topic = country
                return facts[country]

        for country, response in facts.items():
            if country in user_input.lower():
                self.last_topic = country
                return response

        return None

    def generate_reply(self, user_input):
        """Combines small talk, facts, and GPT-2 responses."""
        # 1. Handle greetings or small talk
        small_talk_reply = self.small_talk(user_input)
        if small_talk_reply:
            return small_talk_reply

        # 2. Handle factual questions
        fact = self.find_fact(user_input)
        if fact:
            return fact

        # 3. Otherwise, use GPT-2 for free-form conversation
        context = "\n".join(self.memory)
        prompt = f"{context}\nUser: {user_input}\nBot:"

        response = self.generator(
            prompt,
            max_new_tokens=60,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            truncation=True,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )[0]["generated_text"]

        bot_reply = response.split("Bot:")[-1].split("User:")[0].strip()
        if not bot_reply:
            bot_reply = "I'm not sure about that."

        self.memory.append(f"User: {user_input}")
        self.memory.append(f"Bot: {bot_reply}")
        return bot_reply

    def chat(self):
        """Main chat loop."""
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() == "/exit":
                print("Exiting chatbot. Goodbye! 👋")
                break
            bot_reply = self.generate_reply(user_input)
            print(f"Bot: {bot_reply}\n")


if __name__ == "__main__":
    Chatbot().chat()
